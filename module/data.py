import torch
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# PyTorch 臨床資料集類別 (Clinical Dataset)
# ==============================================================================
class ClinicalDataset(Dataset):
    """
    臨床病患歷史用藥與回診紀錄資料集。
    將連續處方箋的領藥歷史轉換為 PyTorch 模型訓練所需要的張量特徵：
    - x_seq: [seq_len, num_meds + 1] (過去 N 次的用藥一熱編碼向量 + 領藥間隔天數)
    - c_idx: 慢性疾病 ID 索引
    - y_meds: [num_meds] 下一次領藥的實際藥物一熱標籤
    - y_gap: 下一次領藥間隔天數 (正規化至 [0, 1])
    """
    def __init__(self, df, all_meds, disease_to_idx, max_gap=100.0, seq_len=10):
        self.df = df.reset_index(drop=True)
        self.all_meds = all_meds
        self.med_to_idx = {m: i for i, m in enumerate(all_meds)}
        self.disease_to_idx = disease_to_idx
        self.max_gap = max_gap
        self.num_meds = len(all_meds)
        self.seq_len = seq_len

    def __len__(self): 
        return len(self.df)

    def encode_meds(self, med_list):
        """ 將藥物名稱列表編碼為一熱 (One-hot) 向量 """
        vec = torch.zeros(self.num_meds)
        for m in med_list:
            if m in self.med_to_idx: 
                vec[self.med_to_idx[m]] = 1.0
        return vec

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        c_idx = torch.tensor(self.disease_to_idx[row['disease']], dtype=torch.long)
        
        x_seq = []
        for visit in row['history_window']:
            med_vec = self.encode_meds(visit['meds'])
            # 正規化用藥天數 (將例如 60 天轉為 0.6)
            duration_val = torch.tensor([visit['m_days'] / self.max_gap], dtype=torch.float32)
            x_seq.append(torch.cat([med_vec, duration_val]))
            
        x_seq = torch.stack(x_seq)
        
        # 對長度不足 seq_len 的序列進行零填充 (Zero Padding)
        pad_len = self.seq_len - x_seq.shape[0]
        if pad_len > 0:
            padding = torch.zeros((pad_len, self.num_meds + 1), dtype=torch.float32)
            x_seq = torch.cat([padding, x_seq], dim=0)
            
        y_meds = self.encode_meds(row['target']['meds'])
        # 正規化目標領藥間隔天數
        y_gap = torch.tensor([row['target']['M_gap'] / self.max_gap], dtype=torch.float32)
        
        return x_seq, c_idx, y_meds, y_gap, row['client_id'], row['target_pharmacy_day']