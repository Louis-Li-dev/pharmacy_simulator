import copy
import math
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multioutput import MultiOutputClassifier
from sklearn.exceptions import ConvergenceWarning
import warnings

def calc_iou_batch(preds_logits, targets, threshold=0.5):
    """
    計算批次藥物預測的交集並集比 (Intersection over Union, IoU) / Jaccard 相似度指標。
    
    參數:
        preds_logits (torch.Tensor): 模型輸出的 logits。
        targets (torch.Tensor): 實際用藥的一熱（One-hot）/二元標籤。
        threshold (float): 判斷是否開立該藥物的信心度門檻（預設 0.5）。
    
    傳回:
        float: 批次內樣本的平均 IoU 值。
    """
    preds_bin = (torch.sigmoid(preds_logits) > threshold).float()
    intersection = (preds_bin * targets).sum(dim=1)
    union = (preds_bin + targets).clamp(0, 1).sum(dim=1)
    return (intersection / (union + 1e-8)).mean().item()


# ==============================================================================
# 1. 基線 LSTM 模型 (Baseline LSTM with Disease Masking)
# ==============================================================================
class BaselineLSTM(nn.Module):
    """
    序列 LSTM 基線模型。
    結合個案過去的領藥歷史與疾病類別嵌入（Disease Embedding），
    預測下一次領藥的：
    1. 藥物組合 (Medication Logits) - 套用疾病用藥遮罩 (Disease Masking)
    2. 下次領藥間隔天數 (Day Gap Prediction)
    """
    def __init__(self, input_dim, num_diseases, num_meds, disease_mask, hidden_dim=64):
        super().__init__()
        # LSTM 擷取領藥時間序列特徵
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        # 疾病類別嵌入層 (Disease ID -> 16維向量)
        self.disease_embed = nn.Embedding(num_diseases, 16)
        # 全連接特徵整合層
        self.fc = nn.Sequential(nn.Linear(hidden_dim + 16, hidden_dim), nn.ReLU())
        # 藥物預測標頭 (Multi-label Logits)
        self.head_meds = nn.Linear(hidden_dim, num_meds)
        # 間隔天數預測標頭 (保持正數 Softplus)
        self.head_gap = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())
        
        # 註冊疾病遮罩為 Buffer，自動隨模型轉移至 GPU/CPU
        self.register_buffer('disease_mask', disease_mask)

    def forward(self, x, c):
        """
        前向傳播。
        x: [batch_size, seq_len, input_dim] 領藥歷史序列
        c: [batch_size] 疾病類別 ID
        """
        _, (hn, _) = self.lstm(x)
        # 拼接 LSTM 最後隱藏狀態與疾病嵌入
        shared = self.fc(torch.cat([hn[-1], self.disease_embed(c)], dim=1))
        
        med_logits = self.head_meds(shared)
        # 套用內部疾病遮罩：將該疾病不可能開立的藥物 logit 設為極大負值 (-1e9)
        med_logits = med_logits.masked_fill(self.disease_mask[c] == 0, -1e9)
            
        gap_pred = self.head_gap(shared)
        return med_logits, gap_pred

    def fit(self, train_loader, epochs=20, lr=0.005, device='cpu'):
        """
        模型訓練流程。
        組合 BCEWithLogitsLoss (藥物) 與 MSELoss (間隔天數)。
        """
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        self.train()
        epoch_losses = []
        for epoch in range(epochs):
            total_loss = 0
            for x, c, ym, yg, _, _ in train_loader:
                x, c, ym, yg = x.to(device), c.to(device), ym.to(device), yg.to(device)
                optimizer.zero_grad()
                m_logits, g_pred = self(x, c)
                # 損失函數：藥物多標籤交叉熵 + 天數均方誤差 (權重加權 20.0)
                loss = F.binary_cross_entropy_with_logits(m_logits, ym) + (F.mse_loss(g_pred, yg) * 20.0)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / max(1, len(train_loader))
            epoch_losses.append(avg_loss)
            if (epoch+1) % 10 == 0 or epoch == epochs - 1: 
                print(f"[Baseline LSTM] Epoch {epoch+1:02d} Loss: {avg_loss:.4f}")
        return epoch_losses

    def predict(self, loader, device='cpu', apply_mask=True):
        """
        對評估資料集進行預測並還原數值尺度。
        """
        self.to(device); self.eval()
        p_meds, t_meds, p_gaps, t_gaps, c_ids, t_days = [], [], [], [], [], []
        with torch.no_grad():
            for x, c, ym, yg, client_id, t_day in loader:
                x, c = x.to(device), c.to(device)
                m, g = self(x, c)
                
                # 動態套用疾病用藥遮罩
                if apply_mask:
                    m = m.masked_fill(self.disease_mask[c] == 0, -1e9)
                    
                p_meds.append(m.cpu()); t_meds.append(ym)
                # 天數數值乘以 100 還原成實際天數
                p_gaps.append(g.cpu() * 100.0); t_gaps.append(yg * 100.0)
                c_ids.extend(client_id); t_days.extend(t_day.numpy())
        return (torch.cat(p_meds), torch.cat(t_meds), torch.cat(p_gaps), torch.cat(t_gaps), np.array(c_ids), np.array(t_days))


# ==============================================================================
# 2. 統計學基線模型 (Statistical ML Baseline: Logistic + Ridge AR)
# ==============================================================================
class StatisticalBaseline:
    """
    傳統統計學/機器學習基線模型。
    - 藥物預測：使用 MultiOutputClassifier (邏輯迴歸 Logistic Regression)
    - 領藥天數預測：使用 Ridge 嶺迴歸 (作為歷史間隔天數的自迴歸 AR 模型)
    """
    def __init__(self, disease_mask):
        self.disease_mask = disease_mask.cpu()
        
        # 藥物預測模型：多輸出獨立邏輯迴歸 (類別平衡比重)
        self.med_model = MultiOutputClassifier(
            LogisticRegression(max_iter=500, class_weight='balanced', solver='liblinear')
        )
        
        # 間隔天數模型：對過去間隔天數做自迴歸 Ridge 回歸
        self.gap_model = Ridge(alpha=1.0)

    def fit(self, train_loader, device='cpu'):
        """ 擬合統計模型 """
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        
        X_meds, y_meds = [], []
        X_gaps, y_gaps = [], []

        for x, c, ym, yg, _, _ in train_loader:
            batch_size = x.size(0)
            
            # 展開歷史序列: [batch_size, seq_len * input_dim]
            x_flat = x.view(batch_size, -1).numpy()
            c_np = c.numpy().reshape(-1, 1)
            
            # 結合歷史序列與疾病類別特徵
            x_med_feat = np.hstack((x_flat, c_np))
            X_meds.append(x_med_feat)
            y_meds.append(ym.numpy())

            # 擷取過去的領藥間隔天數作為 AR 輸入
            past_gaps = x[:, :, 0].numpy()
            X_gaps.append(past_gaps)
            y_gaps.append(yg.numpy())

        # 轉為 Scikit-learn 矩陣格式
        X_meds = np.vstack(X_meds)
        y_meds = np.vstack(y_meds)
        X_gaps = np.vstack(X_gaps)
        y_gaps = np.concatenate(y_gaps)

        print("[Statistical Baseline] 訓練多輸出邏輯迴歸 (藥物預測)...")
        self.med_model.fit(X_meds, y_meds)

        print("[Statistical Baseline] 訓練 Ridge 自迴歸模型 (天數預測)...")
        self.gap_model.fit(X_gaps, y_gaps)
        print("[Statistical Baseline] 訓練完成。")
    
    def predict(self, loader, device='cpu', apply_mask=True):
        """ 統計模型推論 """
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        
        p_meds, t_meds, p_gaps, t_gaps, c_ids, t_days = [], [], [], [], [], []

        for x, c, ym, yg, client_id, t_day in loader:
            batch_size = x.size(0)
            
            x_flat = x.view(batch_size, -1).numpy()
            c_np = c.numpy().reshape(-1, 1)
            x_med_feat = np.hstack((x_flat, c_np))
            past_gaps = x[:, :, 0].numpy()
            
            # 預測機率
            pred_meds_prob = self.med_model.predict_proba(x_med_feat)
            
            med_probs = np.zeros((batch_size, len(pred_meds_prob)))
            for i, class_probs in enumerate(pred_meds_prob):
                med_probs[:, i] = class_probs[:, 1] if class_probs.shape[1] > 1 else 0.0

            # 轉回 Logit 空間以相容 PyTorch 介面
            med_probs_clipped = np.clip(med_probs, 1e-7, 1 - 1e-7)
            med_logits = np.log(med_probs_clipped / (1 - med_probs_clipped))
            med_logits_tensor = torch.tensor(med_logits, dtype=torch.float32)

            # 套用疾病用藥遮罩
            if apply_mask:
                med_logits_tensor = med_logits_tensor.masked_fill(self.disease_mask[c] == 0, -1e9)
            
            # 預測間隔天數
            pred_gaps = self.gap_model.predict(past_gaps)
            pred_gaps_tensor = torch.tensor(pred_gaps, dtype=torch.float32)

            p_meds.append(med_logits_tensor)
            t_meds.append(ym)
            p_gaps.append(pred_gaps_tensor * 100.0) 
            t_gaps.append(yg * 100.0)
            c_ids.extend(client_id)
            t_days.extend(t_day.numpy())

        return (torch.cat(p_meds), torch.cat(t_meds), torch.cat(p_gaps), torch.cat(t_gaps), np.array(c_ids), np.array(t_days))


# ==============================================================================
# 3. 條件式變異自編碼器模型 (CVAE Generative Model with Focal Loss & Skip Connections)
# ==============================================================================
class VAEGenerativeModel(nn.Module):
    """
    條件式變異自編碼器 (Conditional VAE) 產生式模型。
    - 結合 Focal Loss 解決藥物標籤稀疏性與正負樣本不平衡問題。
    - 解碼器具備殘差跳躍連接 (Skip Connection) 保留時間序列記憶。
    - 支援多重隨機採樣集成 (Ensemble Sampling) 以提供穩定機率預測。
    """
    def __init__(self, input_dim, num_diseases, num_meds, disease_mask, hidden_dim=64, latent_dim=32):
        super().__init__()
        self.disease_mask = disease_mask
        self.register_buffer('mask', disease_mask)
        
        # 疾病條件嵌入
        self.disease_embed = nn.Embedding(num_diseases, 16)
        # 編碼器 LSTM (串接輸入與疾病嵌入)
        self.encoder_lstm = nn.LSTM(input_dim + 16, hidden_dim, batch_first=True)
        # 潛在空間 (Latent Space) 均值與對數方差投影
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # 解碼器 (Decoder)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.head_meds = nn.Linear(hidden_dim, num_meds)
        self.head_gap = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())

    def focal_loss(self, logits, targets, alpha=0.25, gamma=2):
        """
        Focal Loss 加權計算，降低容易分類的負樣本比重，關注難預測的正確藥物。
        """
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        return (alpha * (1 - pt)**gamma * bce).mean()

    def forward(self, x, c):
        """
        CVAE 前向傳播與重參數化技巧 (Reparameterization Trick)。
        """
        # 將疾病嵌入擴展並拼接至歷史序列的每一時間步
        emb = self.disease_embed(c).unsqueeze(1).repeat(1, x.size(1), 1)
        x_in = torch.cat([x, emb], dim=2)
        
        _, (hn, _) = self.encoder_lstm(x_in)
        mu, logvar = self.fc_mu(hn[-1]), self.fc_var(hn[-1])
        # 重參數化採樣 z
        z = mu + torch.randn_like(logvar) * torch.exp(0.5 * logvar)
        
        # 解碼器輸入拼接潛在變數 z 與疾病條件
        dec_in = torch.cat([z, self.disease_embed(c)], dim=1)
        shared = self.decoder(dec_in)
        
        # Skip connection: 結合潛在空間特徵與時間序列記憶
        shared = shared + hn[-1] 
        return self.head_meds(shared), self.head_gap(shared), mu, logvar

    def fit(self, train_loader, epochs=20, lr=0.001, device='cpu'):
        """
        CVAE 訓練流程：包含 Focal Loss + 天數 MSE + KL 散度正規化。
        """
        self.to(device); optimizer = optim.Adam(self.parameters(), lr=lr)
        self.train()
        for epoch in range(epochs):
            for x, c, ym, yg, _, _ in train_loader:
                x, c, ym, yg = x.to(device), c.to(device), ym.to(device), yg.to(device)
                optimizer.zero_grad()
                m_logits, g_pred, mu, logvar = self(x, c)
                
                # 損失組合：50 * Focal Loss (藥物) + 10 * MSE (天數) + 0.001 * KLD
                loss = (self.focal_loss(m_logits, ym) * 50.0) + (F.mse_loss(g_pred, yg) * 10.0) + \
                       (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * 0.001)
                loss.backward(); optimizer.step()

    def predict(self, loader, device='cpu', gen_num=10, apply_mask=True):
        """
        使用多重採樣集成 (Ensemble Sampling across gen_num) 進行推論。
        """
        self.to(device); self.eval()
        p_meds, t_meds, p_gaps, t_gaps, c_ids, t_days = [], [], [], [], [], []
        with torch.no_grad():
            for x, c, ym, yg, client_id, t_day in loader:
                x, c = x.to(device), c.to(device)
                
                # 集成採樣：進行 gen_num 次多重推論並求平均
                all_probs = []
                all_gaps = []
                for _ in range(gen_num):
                    m_logits, g_pred, _, _ = self(x, c)
                    all_probs.append(torch.sigmoid(m_logits))
                    all_gaps.append(g_pred)
                
                avg_probs = torch.stack(all_probs).mean(dim=0)
                g_pred = torch.stack(all_gaps).mean(dim=0)
                
                # 轉回 Logit 空間以保持介面一致
                med_logits = torch.log(avg_probs / (1 - avg_probs + 1e-7) + 1e-7)
                
                if apply_mask:
                    med_logits = med_logits.masked_fill(self.mask[c] == 0, -1e9)
                
                p_meds.append(med_logits.cpu()); t_meds.append(ym)
                p_gaps.append(g_pred.cpu() * 100.0); t_gaps.append(yg * 100.0)
                c_ids.extend(client_id); t_days.extend(t_day.numpy())
        return (torch.cat(p_meds), torch.cat(t_meds), torch.cat(p_gaps), torch.cat(t_gaps), np.array(c_ids), np.array(t_days))