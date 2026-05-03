import torch
from torch.utils.data import Dataset, DataLoader
# ==========================================
# 3. PYTORCH DATASET & DATALOADERS
# ==========================================
class ClinicalDataset(Dataset):
    def __init__(self, df, all_meds, disease_to_idx, max_gap=100.0, seq_len=10):
        self.df = df.reset_index(drop=True)
        self.all_meds = all_meds
        self.med_to_idx = {m: i for i, m in enumerate(all_meds)}
        self.disease_to_idx = disease_to_idx
        self.max_gap = max_gap
        self.num_meds = len(all_meds)
        self.seq_len = seq_len

    def __len__(self): return len(self.df)

    def encode_meds(self, med_list):
        vec = torch.zeros(self.num_meds)
        for m in med_list:
            if m in self.med_to_idx: vec[self.med_to_idx[m]] = 1.0
        return vec

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        c_idx = torch.tensor(self.disease_to_idx[row['disease']], dtype=torch.long)
        
        x_seq = []
        for visit in row['history_window']:
            med_vec = self.encode_meds(visit['meds'])
            # FIX: Properly scale the supply duration (m_days) so 60 becomes 0.6
            duration_val = torch.tensor([visit['m_days'] / self.max_gap], dtype=torch.float32)
            x_seq.append(torch.cat([med_vec, duration_val]))
            
        x_seq = torch.stack(x_seq)
        
        # Zero Padding for sequences shorter than seq_len
        pad_len = self.seq_len - x_seq.shape[0]
        if pad_len > 0:
            padding = torch.zeros((pad_len, self.num_meds + 1), dtype=torch.float32)
            x_seq = torch.cat([padding, x_seq], dim=0)
            
        y_meds = self.encode_meds(row['target']['meds'])
        # Target Gap is also scaled
        y_gap = torch.tensor([row['target']['M_gap'] / self.max_gap], dtype=torch.float32)
        
        return x_seq, c_idx, y_meds, y_gap, row['client_id'], row['target_pharmacy_day']