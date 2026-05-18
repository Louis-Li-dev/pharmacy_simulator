import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
try:
    from scipy.stats import skewnorm
except ImportError:
    class skewnorm:
        @staticmethod
        def rvs(a, loc=0, scale=1):
            return np.random.normal(loc, scale)


class RetailSimulationPipeline:
    def __init__(self, n_transactions=5000, seed=42, max_basket_items=4):
        self.n_transactions = n_transactions
        self.seed = seed
        self.max_basket_items = max_basket_items
        self.rng = np.random.default_rng(seed)
        self.disease_meds = {
            "Hypertension": ["Lisinopril", "Amlodipine", "HCTZ"],
            "Type_2_Diabetes": ["Metformin", "Glipizide", "Empagliflozin", "Insulin"],
            "Asthma": ["Inhaled Corticosteroid", "Albuterol Rescue", "Leukotriene Modifier"],
            "Hyperlipidemia": ["Atorvastatin", "Ezetimibe", "PCSK9_Inhibitor"],
            "Rheumatoid_Arthritis": ["Methotrexate", "Folic_Acid", "Adalimumab"],
        }
        self.disease_names = list(self.disease_meds.keys())
        self.products = [
            "Snacks",
            "Condoms",
            "Cosmetics",
            "Shampoo",
            "Milk_Powder",
            "Bandages",
            "Supplements",
            "Coffee",
            "Adult_Diapers",
            "Sleep_Aids",
        ]
        self.med_aliases = {
            "Hypertension_Meds": self.disease_meds["Hypertension"],
            "Diabetes_Meds": self.disease_meds["Type_2_Diabetes"],
            "Dementia_Meds": ["Supplements", "Sleep_Aids"],
            "Asthma_Inhaler": self.disease_meds["Asthma"],
        }
        self.meds = sorted({m for meds in self.disease_meds.values() for m in meds} | set(self.med_aliases.keys()))
        self.transactions = None

    def _prob_decay(self, age, start, end, max_p, min_p):
        if age <= start:
            return max_p
        if age >= end:
            return min_p
        return max_p - ((age - start) / (end - start)) * (max_p - min_p)

    def _prob_grow(self, age, start, end, min_p, max_p):
        if age <= start:
            return min_p
        if age >= end:
            return max_p
        return min_p + ((age - start) / (end - start)) * (max_p - min_p)

    def _prob_bell(self, age, peak, sigma, max_p):
        return max_p * np.exp(-0.5 * ((age - peak) / sigma) ** 2)

    def _product_probabilities(self, age, disease):
        probs = {
            "Snacks": self._prob_decay(age, 18, 40, 0.80, 0.10),
            "Condoms": self._prob_decay(age, 18, 45, 0.60, 0.00),
            "Cosmetics": self._prob_decay(age, 18, 55, 0.50, 0.10),
            "Coffee": self._prob_bell(age, 35, 15, 0.60),
            "Shampoo": 0.20,
            "Supplements": self._prob_grow(age, 25, 70, 0.10, 0.70),
            "Milk_Powder": self._prob_grow(age, 50, 80, 0.00, 0.50),
            "Bandages": self._prob_grow(age, 40, 80, 0.05, 0.20),
            "Adult_Diapers": self._prob_grow(age, 65, 90, 0.00, 0.60),
            "Sleep_Aids": self._prob_bell(age, 50, 15, 0.30),
        }
        modifiers = {
            "Hypertension": {"Coffee": -0.40, "Sleep_Aids": 0.30, "Supplements": 0.20},
            "Type_2_Diabetes": {"Snacks": -0.60, "Bandages": 0.50, "Milk_Powder": 0.30},
            "Asthma": {"Cosmetics": -0.30, "Coffee": 0.10},
            "Hyperlipidemia": {"Supplements": 0.40},
            "Rheumatoid_Arthritis": {"Bandages": 0.40, "Sleep_Aids": 0.30, "Supplements": 0.20},
        }
        for product, delta in modifiers.get(disease, {}).items():
            probs[product] = float(np.clip(probs[product] + delta, 0.0, 1.0))
        return probs

    def generate_transactions(self):
        records = []
        for tx_id in range(self.n_transactions):
            age_group = self.rng.choice(["young", "middle", "elderly"], p=[0.15, 0.40, 0.45])
            if age_group == "young":
                age = int(self.rng.integers(18, 31))
            elif age_group == "middle":
                age = int(np.clip(self.rng.normal(45, 7), 31, 60))
            else:
                age = int(np.clip(self.rng.normal(72, 9), 55, 95))

            disease = str(self.rng.choice(self.disease_names))
            med_pool = self.disease_meds[disease]
            med_count = int(self.rng.integers(1, min(3, len(med_pool)) + 1))
            meds = sorted(self.rng.choice(med_pool, size=med_count, replace=False).tolist())
            product_probs = self._product_probabilities(age, disease)
            products = [p for p, prob in product_probs.items() if self.rng.binomial(1, prob) == 1]
            if len(products) > self.max_basket_items:
                products = sorted(products, key=lambda p: product_probs[p], reverse=True)[: self.max_basket_items]
            if not products:
                products = [max(product_probs, key=product_probs.get)]

            records.append(
                {
                    "transaction_id": f"T_{tx_id + 1:06d}",
                    "age": age,
                    "Disease": disease,
                    "meds": meds,
                    "products": products,
                }
            )

        self.transactions = pd.DataFrame(records)
        return self.transactions

    def generate_relationship_table(self):
        if self.transactions is None:
            self.generate_transactions()
        matrix = pd.DataFrame(0, index=self.meds, columns=self.products)
        for _, row in self.transactions.iterrows():
            meds = self.expand_meds(row["meds"])
            for med in meds:
                if med not in matrix.index:
                    continue
                for product in row["products"]:
                    matrix.loc[med, product] += 1
        return matrix

    def expand_meds(self, meds_list):
        expanded = []
        for med in meds_list:
            if med in self.med_aliases:
                expanded.extend(self.med_aliases[med])
            else:
                expanded.append(med)
        return sorted(set(expanded))

    def prepare_tensors(self):
        if self.transactions is None:
            self.generate_transactions()
        med_to_idx = {med: idx for idx, med in enumerate(self.meds)}
        prod_to_idx = {product: idx for idx, product in enumerate(self.products)}
        X = torch.zeros((len(self.transactions), 1 + len(self.meds)), dtype=torch.float32)
        Y = torch.zeros((len(self.transactions), len(self.products)), dtype=torch.float32)
        for row_idx, row in self.transactions.iterrows():
            X[row_idx, 0] = float(row["age"]) / 100.0
            for med in self.expand_meds(row["meds"]):
                if med in med_to_idx:
                    X[row_idx, 1 + med_to_idx[med]] = 1.0
            for product in row["products"]:
                Y[row_idx, prod_to_idx[product]] = 1.0
        return X, Y, med_to_idx, prod_to_idx

    def to_diffusion_frame(self):
        if self.transactions is None:
            self.generate_transactions()
        rows = []
        for _, row in self.transactions.iterrows():
            out = {"age": int(row["age"]), "Disease": row["Disease"]}
            for product in self.products:
                out[product] = 1.0 if product in row["products"] else 0.0
            rows.append(out)
        return pd.DataFrame(rows)


class BasketDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class ProductRecommenderNN(nn.Module):
    def __init__(self, input_dim, num_products, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_products),
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 模組 1：資料生成器 (Data Generator)
# ==========================================
class GradientContextSimulator:
    def __init__(self, n_samples=10000):
        self.n_samples = n_samples
        
        # 1. 擴充為 5 大慢性疾病
        self.diseases = [
            "Hypertension (高血壓)", 
            "Type_2_Diabetes (第二型糖尿病)", 
            "Asthma (氣喘)", 
            "Hyperlipidemia (高血脂)", 
            "Rheumatoid_Arthritis (風濕關節炎)"
        ]
        
        # 2. 零售商品庫維持不變 (Target Y)
        self.retail_products = [
            "Snacks (零食)", "Condoms (保險套)", "Cosmetics (化妝品)", 
            "Shampoo (洗髮精)", "Milk_Powder (成人奶粉)", "Bandages (傷用包紮)", 
            "Supplements (保健食品)", "Coffee (咖啡)", "Adult_Diapers (成人紙尿褲)",
            "Sleep_Aids (助眠品)"
        ]

    def prob_decay(self, age, start, end, max_p, min_p):
        if age <= start: return max_p
        if age >= end: return min_p
        return max_p - ((age - start) / (end - start)) * (max_p - min_p)

    def prob_grow(self, age, start, end, min_p, max_p):
        if age <= start: return min_p
        if age >= end: return max_p
        return min_p + ((age - start) / (end - start)) * (max_p - min_p)

    def prob_bell(self, age, peak, sigma, max_p):
        return max_p * np.exp(-0.5 * ((age - peak) / sigma)**2)

    def generate_data(self):
        records = []
        for _ in range(self.n_samples):
            # 決定病患年齡層
            group = np.random.choice(['young', 'middle', 'elderly'], p=[0.15, 0.40, 0.45])
            if group == 'young': raw_age = np.clip(skewnorm.rvs(a=4, loc=18, scale=6), 18, 30)
            elif group == 'middle': raw_age = np.random.normal(45, 7)
            else: raw_age = np.clip(skewnorm.rvs(a=-3, loc=75, scale=10), 55, 95)
            
            age = int(np.round(raw_age))
            disease = np.random.choice(self.diseases)
            
            # 基礎購物機率 (隨年齡變化)
            p_items = {
                "Snacks (零食)": self.prob_decay(age, 18, 40, 0.8, 0.1),
                "Condoms (保險套)": self.prob_decay(age, 18, 45, 0.6, 0.0),
                "Cosmetics (化妝品)": self.prob_decay(age, 18, 55, 0.5, 0.1),
                "Coffee (咖啡)": self.prob_bell(age, 35, 15, 0.6),
                "Shampoo (洗髮精)": 0.2, # 日用品基線
                "Supplements (保健食品)": self.prob_grow(age, 25, 70, 0.1, 0.7),
                "Milk_Powder (成人奶粉)": self.prob_grow(age, 50, 80, 0.0, 0.5),
                "Bandages (傷用包紮)": self.prob_grow(age, 40, 80, 0.05, 0.2),
                "Adult_Diapers (成人紙尿褲)": self.prob_grow(age, 65, 90, 0.0, 0.6),
                "Sleep_Aids (助眠品)": self.prob_bell(age, 50, 15, 0.3)
            }
            
            # ==========================================
            # 💊 全新醫藥情境邏輯 (Contextual Modifiers)
            # ==========================================
            if disease == "Hypertension (高血壓)":
                # 衛教：高血壓需減少咖啡因，並關注睡眠品質
                p_items["Coffee (咖啡)"] = max(0.0, p_items["Coffee (咖啡)"] - 0.4) 
                p_items["Sleep_Aids (助眠品)"] = min(1.0, p_items["Sleep_Aids (助眠品)"] + 0.3)
                p_items["Supplements (保健食品)"] = min(1.0, p_items["Supplements (保健食品)"] + 0.2)

            elif disease == "Type_2_Diabetes (第二型糖尿病)":
                # 衛教：嚴格控糖，並極度注重末梢神經與傷口照護 (糖尿病足)
                p_items["Snacks (零食)"] = max(0.0, p_items["Snacks (零食)"] - 0.6)
                p_items["Bandages (傷用包紮)"] = min(1.0, p_items["Bandages (傷用包紮)"] + 0.5) 
                p_items["Milk_Powder (成人奶粉)"] = min(1.0, p_items["Milk_Powder (成人奶粉)"] + 0.3) # 糖尿病專用奶粉

            elif disease == "Asthma (氣喘)":
                # 衛教：避免化學香精刺激氣管，咖啡因有輕微擴張支氣管效果
                p_items["Cosmetics (化妝品)"] = max(0.0, p_items["Cosmetics (化妝品)"] - 0.3)
                p_items["Coffee (咖啡)"] = min(1.0, p_items["Coffee (咖啡)"] + 0.1)

            elif disease == "Hyperlipidemia (高血脂)":
                # 衛教：心血管保養 (魚油、紅麴等)
                p_items["Supplements (保健食品)"] = min(1.0, p_items["Supplements (保健食品)"] + 0.4)

            elif disease == "Rheumatoid_Arthritis (風濕關節炎)":
                # 衛教：關節疼痛需護具/包紮，且慢性痛易影響睡眠
                p_items["Bandages (傷用包紮)"] = min(1.0, p_items["Bandages (傷用包紮)"] + 0.4)
                p_items["Sleep_Aids (助眠品)"] = min(1.0, p_items["Sleep_Aids (助眠品)"] + 0.3)
                p_items["Supplements (保健食品)"] = min(1.0, p_items["Supplements (保健食品)"] + 0.2) # 關節保養品
            
            # ==========================================
            # 轉換為購物籃 (限制 1~4 件)
            # ==========================================
            basket = {'age': age, 'Disease': disease}
            for item in self.retail_products: basket[item] = 0.0
                
            drawn_items = [item for item, prob in p_items.items() if np.random.binomial(n=1, p=prob) == 1]
            if len(drawn_items) > 4: 
                drawn_items = sorted(drawn_items, key=lambda k: p_items[k], reverse=True)[:4]
            elif len(drawn_items) == 0: 
                drawn_items = [max(p_items.keys(), key=lambda k: p_items[k])]
                
            for item in drawn_items: basket[item] = 1.0
            records.append(basket)
            
        return pd.DataFrame(records)

# ==========================================
# 模組 2：共用評估函數 (Evaluation)
# ==========================================
def evaluate_metrics_dynamic(y_true, y_pred_probs, threshold=0.5):
    """計算 Precision, Recall, Jaccard (限制推薦長度 1~4)"""
    precisions, recalls, jaccards = [], [], []
    
    for i in range(y_true.size(0)):
        true_idx = set(torch.where(y_true[i] == 1)[0].tolist())
        pred_idx = torch.where(y_pred_probs[i] > threshold)[0].tolist()
        
        if len(pred_idx) > 4:
            _, top4 = torch.topk(y_pred_probs[i], 4)
            pred_idx = set(top4.tolist())
        elif len(pred_idx) == 0:
            _, top1 = torch.topk(y_pred_probs[i], 1)
            pred_idx = set(top1.tolist())
        else:
            pred_idx = set(pred_idx)
            
        intersection = len(true_idx.intersection(pred_idx))
        union = len(true_idx.union(pred_idx))
        
        precisions.append(intersection / len(pred_idx) if len(pred_idx) > 0 else 0)
        recalls.append(intersection / len(true_idx) if len(true_idx) > 0 else 0)
        jaccards.append(intersection / union if union > 0 else 0)
        
    return np.mean(precisions), np.mean(recalls), np.mean(jaccards)

def print_evaluation_report(model_name, precision, recall, jaccard):
    print("="*45)
    print(f"🏆 {model_name} 評估結果 (測試集未看過資料)")
    print("="*45)
    print(f"🔹 精準率 (Precision) : {precision*100:.1f}%")
    print(f"🔹 召回率 (Recall)    : {recall*100:.1f}%")
    print(f"🔹 Jaccard 相似度     : {jaccard:.4f}")
    print("="*45)

# ==========================================
# 模組 3：Diffusion 網路架構
# ==========================================
class ContextEmbedding(nn.Module):
    def __init__(self, num_diseases, num_ages=120, emb_dim=32):
        super().__init__()
        self.disease_emb = nn.Embedding(num_embeddings=num_diseases, embedding_dim=emb_dim)
        self.age_emb = nn.Embedding(num_embeddings=num_ages, embedding_dim=emb_dim)
        self.fuse = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim), nn.SiLU(), nn.LayerNorm(emb_dim)
        )

    def forward(self, age_idx, disease_idx):
        d_emb = self.disease_emb(disease_idx)
        a_emb = self.age_emb(age_idx)
        return self.fuse(torch.cat([a_emb, d_emb], dim=1))

class ConditionalDenoisingMLP(nn.Module):
    def __init__(self, y_dim, num_diseases, context_dim=32, time_dim=32):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(1, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.context_mlp = ContextEmbedding(num_diseases=num_diseases, emb_dim=context_dim)
        self.net = nn.Sequential(
            nn.Linear(y_dim + time_dim + context_dim, 128), nn.SiLU(), nn.LayerNorm(128),
            nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, y_dim)
        )

    def forward(self, y_t, age, disease_idx, t):
        t_emb = self.time_mlp(t)
        c_emb = self.context_mlp(age, disease_idx)
        return self.net(torch.cat([y_t, t_emb, c_emb], dim=1))

class TabularDDPM(nn.Module):
    def __init__(self, y_dim, num_diseases, n_steps=50):
        super().__init__()
        self.n_steps = n_steps
        self.model = ConditionalDenoisingMLP(y_dim, num_diseases)
        self.beta = torch.linspace(0.0001, 0.02, n_steps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def forward_noise(self, y0, t, noise):
        alpha_bar_t = self.alpha_bar[t].view(-1, 1)
        return torch.sqrt(alpha_bar_t) * y0 + torch.sqrt(1 - alpha_bar_t) * noise

    def sample(self, age, disease_idx):
        self.model.eval()
        batch_size = age.shape[0]
        device = age.device
        y_t = torch.randn(batch_size, self.model.net[-1].out_features).to(device)
        
        with torch.no_grad():
            for i in reversed(range(self.n_steps)):
                t = torch.full((batch_size, 1), i, dtype=torch.float32).to(device)
                pred_noise = self.model(y_t, age, disease_idx, t / self.n_steps)
                alpha_t, alpha_bar_t = self.alpha[i], self.alpha_bar[i]
                noise = torch.randn_like(y_t) if i > 0 else torch.zeros_like(y_t)
                y_t = (1 / torch.sqrt(alpha_t)) * (y_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * pred_noise) + torch.sqrt(self.beta[i]) * noise
                
        return (y_t + 1.0) / 2.0

# ==========================================
# 模組 4：封裝訓練函數 (Trainers)
# ==========================================
def train_diffusion(train_df, products, num_diseases, config):
    print(f"\n🚀 啟動擴散模型訓練 (Epochs: {config['diff_epochs']}, Steps: {config['diff_steps']})...")
    
    # 資料準備
    age_tensor = torch.tensor(train_df['age'].values, dtype=torch.long)
    disease_tensor = torch.tensor(train_df['Disease_Encoded'].values, dtype=torch.long)
    Y_true = torch.tensor(train_df[products].values, dtype=torch.float32)
    Y_scaled = Y_true * 2.0 - 1.0 
    
    dataset = TensorDataset(age_tensor, disease_tensor, Y_scaled, Y_true)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    # 初始化模型
    y_dim = Y_scaled.shape[1]
    ddpm = TabularDDPM(y_dim=y_dim, num_diseases=num_diseases, n_steps=config['diff_steps'])
    optimizer = optim.Adam(ddpm.parameters(), lr=config['lr'])
    loss_fn = nn.MSELoss()

    start_time = time.time()
    for epoch in range(config['diff_epochs']):
        ddpm.train()
        total_loss = 0
        for age_batch, disease_batch, y_scaled_batch, _ in dataloader:
            optimizer.zero_grad()
            t = torch.randint(0, ddpm.n_steps, (age_batch.size(0),))
            noise = torch.randn_like(y_scaled_batch)
            y_t = ddpm.forward_noise(y_scaled_batch, t, noise)
            
            t_normalized = (t.float() / ddpm.n_steps).unsqueeze(1)
            pred_noise = ddpm.model(y_t, age_batch, disease_batch, t_normalized)
            
            loss = loss_fn(pred_noise, noise)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"   Epoch [{epoch+1}/{config['diff_epochs']}], Loss: {total_loss/len(dataloader):.4f}")
            
    print(f"✅ Diffusion 訓練完成！耗時: {time.time() - start_time:.2f} 秒")
    return ddpm

def train_random_forest(train_df, products, config):
    print(f"\n🌲 啟動隨機森林訓練 (Trees: {config['rf_estimators']}, Depth: {config['rf_max_depth']})...")
    X_train = pd.get_dummies(train_df[['age', 'Disease']], columns=['Disease']).values
    Y_train = train_df[products].values

    rf_model = RandomForestClassifier(
        n_estimators=config['rf_estimators'], 
        max_depth=config['rf_max_depth'], 
        random_state=42, n_jobs=-1
    )
    
    start_time = time.time()
    rf_model.fit(X_train, Y_train)
    print(f"✅ RF 訓練完成！耗時: {time.time() - start_time:.2f} 秒")
    return rf_model

# ==========================================
# 主程式 (Main Pipeline)
# ==========================================
if __name__ == "__main__":
    # --- 0. 設定所有超參數 ---
    config = {
        "n_samples": 10000,
        "test_size": 0.2,
        "batch_size": 128,
        "lr": 1e-3,
        "diff_epochs": 100,
        "diff_steps": 50,
        "rf_estimators": 100,
        "rf_max_depth": 15,
        "eval_threshold": 0.5
    }

    # --- 1. 生成與切分資料 ---
    print("產生模擬資料中...")
    simulator = GradientContextSimulator(n_samples=config['n_samples'])
    df = simulator.generate_data()
    products = simulator.retail_products
    
    # 全局 Label Encoding 以確保 Train/Test 對齊
    encoder = LabelEncoder()
    df['Disease_Encoded'] = encoder.fit_transform(df['Disease'])
    num_diseases = len(encoder.classes_)
    
    # 嚴謹的 Train-Test Split (特徵與目標一起切)
    train_df, test_df = train_test_split(df, test_size=config['test_size'], random_state=42)
    print(f"資料準備完畢。訓練集: {len(train_df)}筆, 測試集: {len(test_df)}筆")

    # --- 2. 訓練與評估 Diffusion Model ---
    ddpm_model = train_diffusion(train_df, products, num_diseases, config)
    
    # 準備測試資料
    test_age = torch.tensor(test_df['age'].values, dtype=torch.long)
    test_disease = torch.tensor(test_df['Disease_Encoded'].values, dtype=torch.long)
    test_Y_true = torch.tensor(test_df[products].values, dtype=torch.float32)
    
    # 採樣評估
    diff_pred_probs = ddpm_model.sample(test_age, test_disease)
    diff_p, diff_r, diff_j = evaluate_metrics_dynamic(test_Y_true, diff_pred_probs, threshold=config['eval_threshold'])
    print_evaluation_report("擴散模型 (Tabular DDPM)", diff_p, diff_r, diff_j)

    # --- 3. 訓練與評估 Random Forest ---
    rf_model = train_random_forest(train_df, products, config)
    
    # 準備測試資料 (需保持 get_dummies 的欄位對齊，此處為簡化寫法)
    X_test_rf = pd.get_dummies(test_df[['age', 'Disease']], columns=['Disease']).reindex(
        columns=pd.get_dummies(train_df[['age', 'Disease']], columns=['Disease']).columns, fill_value=0
    ).values
    
    # 推論評估
    rf_proba_list = rf_model.predict_proba(X_test_rf)
    rf_pred_probs_np = np.column_stack([probs[:, 1] for probs in rf_proba_list])
    rf_pred_probs = torch.tensor(rf_pred_probs_np, dtype=torch.float32)
    
    rf_p, rf_r, rf_j = evaluate_metrics_dynamic(test_Y_true, rf_pred_probs, threshold=config['eval_threshold'])
    print_evaluation_report("隨機森林 (Random Forest)", rf_p, rf_r, rf_j)
