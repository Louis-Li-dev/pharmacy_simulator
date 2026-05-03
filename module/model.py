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
    preds_bin = (torch.sigmoid(preds_logits) > threshold).float()
    intersection = (preds_bin * targets).sum(dim=1)
    union = (preds_bin + targets).clamp(0, 1).sum(dim=1)
    return (intersection / (union + 1e-8)).mean().item()

# --- 1. MASKED BASELINE LSTM ---
class BaselineLSTM(nn.Module):
    def __init__(self, input_dim, num_diseases, num_meds, disease_mask, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.disease_embed = nn.Embedding(num_diseases, 16)
        self.fc = nn.Sequential(nn.Linear(hidden_dim + 16, hidden_dim), nn.ReLU())
        self.head_meds = nn.Linear(hidden_dim, num_meds)
        self.head_gap = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())
        
        # Register mask as a buffer so it moves to device automatically
        self.register_buffer('disease_mask', disease_mask)

    def forward(self, x, c):
        _, (hn, _) = self.lstm(x)
        shared = self.fc(torch.cat([hn[-1], self.disease_embed(c)], dim=1))
        
        med_logits = self.head_meds(shared)
        # Apply the internal mask
        med_logits = med_logits.masked_fill(self.disease_mask[c] == 0, -1e9)
            
        gap_pred = self.head_gap(shared)
        return med_logits, gap_pred

    def fit(self, train_loader, epochs=20, lr=0.005, device='cpu'):
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for x, c, ym, yg, _, _ in train_loader:
                x, c, ym, yg = x.to(device), c.to(device), ym.to(device), yg.to(device)
                optimizer.zero_grad()
                m_logits, g_pred = self(x, c)
                loss = F.binary_cross_entropy_with_logits(m_logits, ym) + (F.mse_loss(g_pred, yg) * 20.0)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch+1) % 10 == 0: 
                print(f"[Baseline LSTM] Epoch {epoch+1:02d} Loss: {total_loss/len(train_loader):.4f}")

    def predict(self, loader, device='cpu', apply_mask=True):
        self.to(device); self.eval()
        p_meds, t_meds, p_gaps, t_gaps, c_ids, t_days = [], [], [], [], [], []
        with torch.no_grad():
            for x, c, ym, yg, client_id, t_day in loader:
                x, c = x.to(device), c.to(device)
                m, g = self(x, c)
                
                # Dynamically apply mask here
                if apply_mask:
                    m = m.masked_fill(self.disease_mask[c] == 0, -1e9)
                    
                p_meds.append(m.cpu()); t_meds.append(ym)
                p_gaps.append(g.cpu() * 100.0); t_gaps.append(yg * 100.0)
                c_ids.extend(client_id); t_days.extend(t_day.numpy())
        return (torch.cat(p_meds), torch.cat(t_meds), torch.cat(p_gaps), torch.cat(t_gaps), np.array(c_ids), np.array(t_days))

# --- 3. STATISTICAL ML BASELINE ---
class StatisticalBaseline:
    def __init__(self, disease_mask):
        """
        Uses Logistic Regression for Disease/Medication mapping
        and Ridge Regression (Auto-Regressive model) for day gap forecasting.
        """
        self.disease_mask = disease_mask.cpu()
        
        # Meds Model: Multi-label Logistic Regression
        self.med_model = MultiOutputClassifier(
            LogisticRegression(max_iter=500, class_weight='balanced', solver='liblinear')
        )
        
        # Gap Model: Ridge regression acting as an Auto-Regressive (AR) model over past gaps
        self.gap_model = Ridge(alpha=1.0)

    def fit(self, train_loader, device='cpu'):
        # Ignore minor convergence warnings from scikit-learn
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        
        X_meds, y_meds = [], []
        X_gaps, y_gaps = [], []

        for x, c, ym, yg, _, _ in train_loader:
            batch_size = x.size(0)
            
            # Flatten the sequence: [batch_size, seq_len * input_dim]
            x_flat = x.view(batch_size, -1).numpy()
            c_np = c.numpy().reshape(-1, 1)
            
            # Combine history + disease label for medication prediction
            x_med_feat = np.hstack((x_flat, c_np))
            X_meds.append(x_med_feat)
            y_meds.append(ym.numpy())

            # For the AR model, extract only the past gaps
            # Assuming gap is the first feature in x: [batch, seq_len, 0]
            past_gaps = x[:, :, 0].numpy()
            X_gaps.append(past_gaps)
            y_gaps.append(yg.numpy())

        # Stack into standard scikit-learn 2D arrays
        X_meds = np.vstack(X_meds)
        y_meds = np.vstack(y_meds)
        X_gaps = np.vstack(X_gaps)
        y_gaps = np.concatenate(y_gaps)

        print("[Statistical Baseline] Fitting Multi-Output Logistic Regression for Meds...")
        self.med_model.fit(X_meds, y_meds)

        print("[Statistical Baseline] Fitting AR (Ridge) model for Day Gaps...")
        self.gap_model.fit(X_gaps, y_gaps)
        print("[Statistical Baseline] Training Complete.")
    
    def predict(self, loader, device='cpu', apply_mask=True):
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        
        p_meds, t_meds, p_gaps, t_gaps, c_ids, t_days = [], [], [], [], [], []

        for x, c, ym, yg, client_id, t_day in loader:
            batch_size = x.size(0)
            
            # Formatting features for prediction
            x_flat = x.view(batch_size, -1).numpy()
            c_np = c.numpy().reshape(-1, 1)
            x_med_feat = np.hstack((x_flat, c_np))
            past_gaps = x[:, :, 0].numpy()
            
            # Predict
            pred_meds_prob = self.med_model.predict_proba(x_med_feat)
            
            # predict_proba returns a list of arrays for multi-output. 
            # We need to stack them into a shape of (batch_size, num_meds)
            med_probs = np.zeros((batch_size, len(pred_meds_prob)))
            for i, class_probs in enumerate(pred_meds_prob):
                # Class 1 probabilities
                med_probs[:, i] = class_probs[:, 1] if class_probs.shape[1] > 1 else 0.0

            # Convert to logits space so it matches PyTorch outputs (inverse sigmoid logic)
            # Clip to prevent log(0)
            med_probs_clipped = np.clip(med_probs, 1e-7, 1 - 1e-7)
            med_logits = np.log(med_probs_clipped / (1 - med_probs_clipped))
            med_logits_tensor = torch.tensor(med_logits, dtype=torch.float32)

            # Apply disease mask
            if apply_mask:
                med_logits_tensor = med_logits_tensor.masked_fill(self.disease_mask[c] == 0, -1e9)
            # Predict gaps (ARIMA equivalent)
            pred_gaps = self.gap_model.predict(past_gaps)
            pred_gaps_tensor = torch.tensor(pred_gaps, dtype=torch.float32)

            # Append results
            p_meds.append(med_logits_tensor)
            t_meds.append(ym)
            p_gaps.append(pred_gaps_tensor * 100.0) 
            t_gaps.append(yg * 100.0)
            c_ids.extend(client_id)
            t_days.extend(t_day.numpy())

        return (torch.cat(p_meds), torch.cat(t_meds), torch.cat(p_gaps), torch.cat(t_gaps), np.array(c_ids), np.array(t_days))
    


class VAEGenerativeModel(nn.Module):
    def __init__(self, input_dim, num_diseases, num_meds, disease_mask, hidden_dim=64, latent_dim=32):
        super().__init__()
        self.disease_mask = disease_mask
        self.register_buffer('mask', disease_mask)
        
        # CVAE: Embed disease in Encoder as well
        self.disease_embed = nn.Embedding(num_diseases, 16)
        self.encoder_lstm = nn.LSTM(input_dim + 16, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder with Skip Connection from latent + disease
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.head_meds = nn.Linear(hidden_dim, num_meds)
        self.head_gap = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())

    def focal_loss(self, logits, targets, alpha=0.25, gamma=2):
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        return (alpha * (1 - pt)**gamma * bce).mean()

    def forward(self, x, c):
        # Concatenate disease embedding to every timestep in input
        emb = self.disease_embed(c).unsqueeze(1).repeat(1, x.size(1), 1)
        x_in = torch.cat([x, emb], dim=2)
        
        _, (hn, _) = self.encoder_lstm(x_in)
        mu, logvar = self.fc_mu(hn[-1]), self.fc_var(hn[-1])
        z = mu + torch.randn_like(logvar) * torch.exp(0.5 * logvar)
        
        dec_in = torch.cat([z, self.disease_embed(c)], dim=1)
        shared = self.decoder(dec_in)
        
        # Skip connection: combine latent features with temporal memory
        shared = shared + hn[-1] 
        return self.head_meds(shared), self.head_gap(shared), mu, logvar

    def fit(self, train_loader, epochs=20, lr=0.001, device='cpu'):
        self.to(device); optimizer = optim.Adam(self.parameters(), lr=lr)
        self.train()
        for epoch in range(epochs):
            for x, c, ym, yg, _, _ in train_loader:
                x, c, ym, yg = x.to(device), c.to(device), ym.to(device), yg.to(device)
                optimizer.zero_grad()
                m_logits, g_pred, mu, logvar = self(x, c)
                
                # Weight Focal Loss higher to focus on medication hits
                loss = (self.focal_loss(m_logits, ym) * 50.0) + (F.mse_loss(g_pred, yg) * 10.0) + \
                       (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * 0.001)
                loss.backward(); optimizer.step()

    def predict(self, loader, device='cpu', gen_num=10, apply_mask=True):
        self.to(device); self.eval()
        p_meds, t_meds, p_gaps, t_gaps, c_ids, t_days = [], [], [], [], [], []
        with torch.no_grad():
            for x, c, ym, yg, client_id, t_day in loader:
                x, c = x.to(device), c.to(device)
                
                # Ensemble: Average probabilities across gen_num
                all_probs = []
                all_gaps = []
                for _ in range(gen_num):
                    m_logits, g_pred, _, _ = self(x, c)
                    all_probs.append(torch.sigmoid(m_logits))
                    all_gaps.append(g_pred)
                
                avg_probs = torch.stack(all_probs).mean(dim=0)
                g_pred = torch.stack(all_gaps).mean(dim=0)
                
                # Apply Thresholding on mean probability
                med_logits = torch.log(avg_probs / (1 - avg_probs + 1e-7) + 1e-7)
                
                if apply_mask:
                    med_logits = med_logits.masked_fill(self.mask[c] == 0, -1e9)
                
                p_meds.append(med_logits.cpu()); t_meds.append(ym)
                p_gaps.append(g_pred.cpu() * 100.0); t_gaps.append(yg * 100.0)
                c_ids.extend(client_id); t_days.extend(t_day.numpy())
        return (torch.cat(p_meds), torch.cat(t_meds), torch.cat(p_gaps), torch.cat(t_gaps), np.array(c_ids), np.array(t_days))