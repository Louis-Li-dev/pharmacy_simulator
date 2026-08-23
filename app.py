"""
==============================================================================
Pharmacy Simulator - 智慧藥局模擬與慢性病用藥/零售預測系統後端 (Flask Server)
==============================================================================
本檔案為系統主要控制中樞，提供以下核心服務：
1. Web RESTful API 路由與動態 HTML 儀表板
2. 多 SQLite 資料庫動態切換、初始化與 CSV 資料匯入/匯出
3. 慢性病連續處方箋用藥預測模型（Baseline LSTM, CVAE, Statistical Baseline）之訓練與推論 API
4. 藥局 OTC / 保健食品連帶銷售推薦模型（ProductRecommenderNN, Tabular DDPM） API
5. 地理資訊服務：結合 OpenStreetMap Overpass API 自動抓取超商/藥局座標並進行個案分派
"""

import base64
import io
import sys
from dataclasses import dataclass

# 重新設定 Windows 環境下的標準輸出編碼為 UTF-8，避免主機文字編碼錯誤
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, Response, jsonify, render_template, request, send_file
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from module.data import ClinicalDataset
from module.model import BaselineLSTM
from module.retail_pipeline import BasketDataset, ProductRecommenderNN, train_diffusion
import fake_locations; from fake_locations import get_client_location
import sqlite3
import os
import math
import requests

# ==============================================================================
# 系統全域記憶體狀態 (Global Application State)
# ==============================================================================
app_state = {
    "pipeline": None,                # 當前已載入之 Data Pipeline 實例
    "train_loader": None,            # PyTorch 訓練 DataLoader
    "test_loader": None,             # PyTorch 測試 DataLoader
    "lstm_model": None,              # 已訓練之 Baseline LSTM 模型
    "retail_model": None,            # 已訓練之零售推薦神經網路
    "retail_diffusion_model": None,  # 已訓練之零售 Tabular DDPM 擴散模型
    "retail_disease_encoder": None,  # 零售模型疾病 LabelEncoder
    "retail_med_to_idx": None,       # 藥物名稱對應索引字典
    "retail_prod_to_idx": None,      # 零售商品名稱對應索引字典
    "retail_products": [],           # 零售商品類別清單
    "results_df": None,              # 預測結果比較表 DataFrame
    "test_uids": [],                 # 測試集個案 Client ID 清單
    "training_log": [],              # 模型訓練歷程 Log 紀錄
    "model_analysis_plot": None,     # 模型表現比對圖表 Base64
    "active_db": "default",          # 當前啟用的 SQLite 資料庫名稱
    "active_model_id": ""            # 當前載入之模型快照識別碼
}

DATABASES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "databases")

def get_db_path():
    """ 取得當前啟用之 SQLite 資料庫檔案路徑 """
    os.makedirs(DATABASES_DIR, exist_ok=True)
    return os.path.join(DATABASES_DIR, f"{app_state.get('active_db', 'default')}.db")


# For backwards compatibility with external scripts, sync default DB on startup
DB_PATH = get_db_path()
fake_locations.set_db_path(DB_PATH)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS convenience_stores (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        osm_id TEXT UNIQUE,
        name TEXT NOT NULL,
        address TEXT,
        name_en TEXT,
        address_en TEXT,
        lat REAL,
        lon REAL
    );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS clinical_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        client_id TEXT,
        age INTEGER,
        disease TEXT,
        doctor_visit_day INTEGER,
        pharmacy_report_day INTEGER,
        m_days INTEGER,
        meds_received TEXT,
        retail_products TEXT,
        assigned_store_id TEXT,
        client_lat REAL,
        client_lon REAL
    );
    """)
    conn.commit()
    conn.close()

# init_db() will be called later after UploadedPipeline is defined

def init_pipeline_from_db():
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='clinical_records'")
            if cursor.fetchone()[0] > 0:
                cursor.execute("SELECT COUNT(*) FROM clinical_records")
                count = cursor.fetchone()[0]
                if count > 0:
                    df = pd.read_sql_query("SELECT * FROM clinical_records", conn)
                    conn.close()
                    # Ensure numeric columns are properly formatted
                    numeric_cols = ["age", "doctor_visit_day", "pharmacy_report_day", "m_days"]
                    for col in numeric_cols:
                        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
                    print(f"Loading pipeline on startup with {len(df)} records from database.")
                    app_state["pipeline"] = UploadedPipeline(df, split_ratio=0.8, seq_len=5)
                else:
                    conn.close()
            else:
                conn.close()
        except Exception as e:
            print(f"Failed to auto-load pipeline from database: {e}")

# init_pipeline_from_db() will be called later after UploadedPipeline is defined

app = Flask(__name__)

REQUIRED_COLUMNS = {
    "client_id",
    "age",
    "disease",
    "doctor_visit_day",
    "pharmacy_report_day",
    "m_days",
    "meds_received",
    "retail_products",
}

def split_items(value):
    return [item.strip() for item in str(value or "").split("|") if item.strip()]


def encode_items(items, mapping):
    vector = torch.zeros(len(mapping), dtype=torch.float32)
    for item in items:
        if item in mapping:
            vector[mapping[item]] = 1.0
    return vector


@dataclass
class UploadedPipeline:
    df: pd.DataFrame
    split_ratio: float = 0.8
    seq_len: int = 5

    def __post_init__(self):
        self.df = self.df.copy()
        self.df["meds_received"] = self.df["meds_received"].apply(split_items)
        self.df["retail_products"] = self.df["retail_products"].apply(split_items)
        numeric_cols = ["age", "doctor_visit_day", "pharmacy_report_day", "m_days"]
        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        self.df = self.df.dropna(subset=numeric_cols)
        self.df[numeric_cols] = self.df[numeric_cols].astype(int)
        self.df = self.df.sort_values(["client_id", "pharmacy_report_day"]).reset_index(drop=True)
        self.split_day = int(self.df["pharmacy_report_day"].max() * self.split_ratio)
        self.disease_names = sorted(self.df["disease"].dropna().unique().tolist())
        self.all_meds = sorted({med for meds in self.df["meds_received"] for med in meds})
        self.products = sorted({product for products in self.df["retail_products"] for product in products})
        self.disease_to_idx = {disease: idx for idx, disease in enumerate(self.disease_names)}
        self.med_to_idx = {med: idx for idx, med in enumerate(self.all_meds)}
        self.prod_to_idx = {product: idx for idx, product in enumerate(self.products)}
        self.sequence_df = None
        self.train_seq_df = None
        self.test_seq_df = None
        self.disease_mask = None
        self.preprocess_sequences(self.seq_len)
        self.create_mappings()

    def preprocess_sequences(self, seq_len=5):
        self.df["prev_report_day"] = self.df.groupby("client_id")["pharmacy_report_day"].shift(1)
        self.df["prev_m_days"] = self.df.groupby("client_id")["m_days"].shift(1)
        self.df["gap"] = self.df["pharmacy_report_day"] - (self.df["prev_report_day"] + self.df["prev_m_days"])
        self.df["gap"] = self.df["gap"].clip(lower=0).fillna(0)

        # Convert to lists/arrays for extremely fast indexing
        client_ids = self.df["client_id"].values
        ages = self.df["age"].values
        diseases = self.df["disease"].values
        report_days = self.df["pharmacy_report_day"].values
        gaps = self.df["gap"].values
        meds = self.df["meds_received"].values
        m_days_arr = self.df["m_days"].values
        products = self.df["retail_products"].values

        sequences = []
        n = len(self.df)
        if n > 0:
            current_client = client_ids[0]
            client_start_idx = 0
            for i in range(1, n):
                if client_ids[i] != current_client:
                    # process current client's records
                    for idx in range(client_start_idx + 1, i):
                        history_start = max(client_start_idx, idx - seq_len)
                        history_window = [
                            {
                                "M_gap": float(gaps[j]),
                                "meds": meds[j],
                                "m_days": int(m_days_arr[j]),
                            }
                            for j in range(history_start, idx)
                        ]
                        sequences.append({
                            "client_id": current_client,
                            "age": int(ages[idx]),
                            "disease": diseases[idx],
                            "target_pharmacy_day": int(report_days[idx]),
                            "prev_pharmacy_day": int(report_days[idx-1]),
                            "history_window": history_window,
                            "target": {
                                "M_gap": float(gaps[idx]),
                                "meds": meds[idx],
                                "products": products[idx],
                            }
                        })
                    current_client = client_ids[i]
                    client_start_idx = i
            
            # process the last client
            for idx in range(client_start_idx + 1, n):
                history_start = max(client_start_idx, idx - seq_len)
                history_window = [
                    {
                        "M_gap": float(gaps[j]),
                        "meds": meds[j],
                        "m_days": int(m_days_arr[j]),
                    }
                    for j in range(history_start, idx)
                ]
                sequences.append({
                    "client_id": current_client,
                    "age": int(ages[idx]),
                    "disease": diseases[idx],
                    "target_pharmacy_day": int(report_days[idx]),
                    "prev_pharmacy_day": int(report_days[idx-1]),
                    "history_window": history_window,
                    "target": {
                        "M_gap": float(gaps[idx]),
                        "meds": meds[idx],
                        "products": products[idx],
                    }
                })

        self.sequence_df = pd.DataFrame(sequences)
        if self.sequence_df.empty:
            self.sequence_df = pd.DataFrame(columns=[
                "client_id", "age", "disease", "target_pharmacy_day", 
                "prev_pharmacy_day", "history_window", "target"
            ])
        self.train_seq_df = self.sequence_df[self.sequence_df["target_pharmacy_day"] < self.split_day].copy()
        self.test_seq_df = self.sequence_df[self.sequence_df["target_pharmacy_day"] >= self.split_day].copy()

    def create_mappings(self):
        mask = torch.zeros((len(self.disease_names), len(self.all_meds)), dtype=torch.float32)
        for disease, group in self.df.groupby("disease"):
            disease_idx = self.disease_to_idx[disease]
            for meds in group["meds_received"]:
                for med in meds:
                    mask[disease_idx, self.med_to_idx[med]] = 1.0
        self.disease_mask = mask


# Execute database initialization & startup loading now that classes are fully defined
init_db()
init_pipeline_from_db()


class RetailVisitDataset(torch.utils.data.Dataset):
    def __init__(self, df, med_to_idx, prod_to_idx):
        self.df = df.reset_index(drop=True)
        self.med_to_idx = med_to_idx
        self.prod_to_idx = prod_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = torch.zeros(1 + len(self.med_to_idx), dtype=torch.float32)
        y = torch.zeros(len(self.prod_to_idx), dtype=torch.float32)
        x[0] = float(row["age"]) / 100.0
        for med in row["meds_received"]:
            if med in self.med_to_idx:
                x[1 + self.med_to_idx[med]] = 1.0
        for product in row["retail_products"]:
            if product in self.prod_to_idx:
                y[self.prod_to_idx[product]] = 1.0
        return x, y


def reset_training_state():
    for key in [
        "train_loader",
        "test_loader",
        "lstm_model",
        "retail_model",
        "retail_diffusion_model",
        "retail_disease_encoder",
        "retail_med_to_idx",
        "retail_prod_to_idx",
        "retail_products",
        "results_df",
        "test_uids",
        "model_analysis_plot",
    ]:
        app_state[key] = [] if key in {"retail_products", "test_uids"} else None
    app_state["training_log"] = []


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_summary(pipeline):
    df = pipeline.df
    disease_counts = df["disease"].value_counts().reset_index()
    disease_counts.columns = ["disease", "count"]
    product_counts = {}
    for products in df["retail_products"]:
        for product in products:
            product_counts[product] = product_counts.get(product, 0) + 1
    top_products = sorted(product_counts.items(), key=lambda item: item[1], reverse=True)[:10]
    return {
        "rows": int(len(df)),
        "clients": int(df["client_id"].nunique()),
        "diseases": int(df["disease"].nunique()),
        "meds": len(pipeline.all_meds),
        "products": len(pipeline.products),
        "train_sequences": int(len(pipeline.train_seq_df)),
        "test_sequences": int(len(pipeline.test_seq_df)),
        "disease_counts": disease_counts.to_dict("records"),
        "top_products": [{"product": product, "count": int(count)} for product, count in top_products],
        "age": {
            "min": int(df["age"].min()),
            "mean": round(float(df["age"].mean()), 1),
            "max": int(df["age"].max()),
        },
    }


def plot_summary(summary):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    diseases = summary["disease_counts"]
    axes[0].bar([item["disease"] for item in diseases], [item["count"] for item in diseases], color="#111827")
    axes[0].set_title("Disease Distribution")
    axes[0].tick_params(axis="x", rotation=35)

    products = summary["top_products"]
    axes[1].barh([item["product"] for item in products][::-1], [item["count"] for item in products][::-1], color="#475569")
    axes[1].set_title("Top Retail Products")
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_model_view_preview(pipeline, sample_size=10):
    sample_clients = pipeline.df["client_id"].drop_duplicates().head(sample_size).tolist()
    sample = pipeline.df[pipeline.df["client_id"].isin(sample_clients)]
    fig, ax = plt.subplots(figsize=(12, 5))
    y_labels = []
    for idx, client_id in enumerate(sample_clients):
        patient = sample[sample["client_id"] == client_id].sort_values("pharmacy_report_day")
        if patient.empty:
            continue
        disease = patient["disease"].iloc[0].replace("_", " ")
        y_labels.append(f"{client_id}\n({disease})")
        for _, row in patient.iterrows():
            rx_day = int(row["pharmacy_report_day"])
            end_day = rx_day + int(row["m_days"])
            ax.plot([rx_day, end_day], [idx, idx], color="#d4d4d8", linewidth=4, solid_capstyle="butt")
            ax.scatter(rx_day, idx, marker="s", color="#111827", s=16, zorder=3)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("Day of Simulation")
    ax.set_title("Model View: Pharmacy Visits and Medication Coverage")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    return fig_to_base64(fig)


def plot_medication_analysis(pipeline, results_df, sample_client=None):
    if sample_client is None:
        sample_client = results_df["client_id"].iloc[0]
    patient_results = results_df[results_df["client_id"] == sample_client].sort_values("target_pharmacy_day").tail(12)
    disease = patient_results["disease"].iloc[0] if not patient_results.empty else ""
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for idx, (_, row) in enumerate(patient_results.iterrows()):
        actual_day = float(row["target_pharmacy_day"])
        pred_day = float(row["predicted_pharmacy_day"])
        ax.plot([actual_day, pred_day], [1, 0], color="#94a3b8", linestyle="--", alpha=0.8)
        ax.scatter(actual_day, 1, color="#16a34a", s=28, zorder=3)
        ax.scatter(pred_day, 0, color="#e11d48", s=28, zorder=3)
        ax.text(actual_day, 1.08, row["true_meds"].replace("|", "\n"), ha="center", va="bottom", fontsize=6,
                bbox=dict(facecolor="white", edgecolor="#86efac", alpha=0.9, pad=2))
        ax.text(pred_day, -0.08, row["predicted_meds"].replace("|", "\n"), ha="center", va="top", fontsize=6,
                bbox=dict(facecolor="white", edgecolor="#fda4af", alpha=0.9, pad=2))
    ax.set_title(f"Medication Prediction Analysis | Client: {sample_client} ({disease})", fontweight="bold", pad=18)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Model Prediction", "Ground Truth"])
    ax.set_xlabel("Timeline (Month)")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_ylim(-0.18, 1.18)
    if not patient_results.empty:
        x_values = patient_results[["target_pharmacy_day", "predicted_pharmacy_day"]].to_numpy().flatten()
        ax.set_xlim(float(np.min(x_values)) - 30, float(np.max(x_values)) + 30)
        ticks = ax.get_xticks()
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{tick / 30:.0f} mo" for tick in ticks])
    fig.subplots_adjust(top=0.82, bottom=0.22, left=0.12, right=0.98)
    return fig_to_base64(fig)


def train_retail_nn(pipeline, epochs=10):
    dataset = RetailVisitDataset(pipeline.df, pipeline.med_to_idx, pipeline.prod_to_idx)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = ProductRecommenderNN(1 + len(pipeline.med_to_idx), len(pipeline.prod_to_idx))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    model.train()
    logs = []
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        logs.append(f"Retail NN epoch {epoch + 1}/{epochs}: loss={total_loss / max(1, len(loader)):.4f}")
    return model, logs


def train_retail_diffusion(pipeline, epochs=10, steps=30):
    diffusion_df = pipeline.df[["age", "disease"]].rename(columns={"disease": "Disease"}).copy()
    for product in pipeline.products:
        diffusion_df[product] = pipeline.df["retail_products"].apply(lambda items: 1.0 if product in items else 0.0)
    encoder = LabelEncoder()
    diffusion_df["Disease_Encoded"] = encoder.fit_transform(diffusion_df["Disease"])
    model, epoch_losses = train_diffusion(
        diffusion_df,
        pipeline.products,
        len(encoder.classes_),
        {"batch_size": 128, "lr": 1e-3, "diff_epochs": epochs, "diff_steps": steps},
    )
    return model, encoder, epoch_losses


def top_meds_from_logits(logits, meds, limit=3):
    probs = torch.sigmoid(logits).detach().cpu().flatten()
    selected = torch.where(probs >= 0.5)[0].tolist()
    if not selected:
        selected = torch.topk(probs, min(limit, len(meds))).indices.tolist()
    selected = sorted(selected, key=lambda idx: float(probs[idx]), reverse=True)[:limit]
    return [meds[idx] for idx in selected]


def recommend_products(age, disease, meds, limit=4):
    scores = {}
    diffusion_model = app_state["retail_diffusion_model"]
    encoder = app_state["retail_disease_encoder"]
    products = app_state["retail_products"]
    if diffusion_model is not None and encoder is not None and disease in set(encoder.classes_):
        with torch.no_grad():
            disease_idx = int(encoder.transform([disease])[0])
            probs = diffusion_model.sample(
                torch.tensor([int(np.clip(age, 0, 119))], dtype=torch.long),
                torch.tensor([disease_idx], dtype=torch.long),
            ).squeeze().cpu().numpy()
        for product, prob in zip(products, probs):
            scores.setdefault(product, []).append(float(prob))

    retail_model = app_state["retail_model"]
    med_to_idx = app_state["retail_med_to_idx"]
    prod_to_idx = app_state["retail_prod_to_idx"]
    if retail_model is not None and med_to_idx is not None and prod_to_idx is not None:
        x = torch.zeros(1 + len(med_to_idx), dtype=torch.float32)
        x[0] = float(age) / 100.0
        for med in meds:
            if med in med_to_idx:
                x[1 + med_to_idx[med]] = 1.0
        retail_model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(retail_model(x.unsqueeze(0))).squeeze().cpu().numpy()
        idx_to_prod = {idx: product for product, idx in prod_to_idx.items()}
        for idx, prob in enumerate(probs):
            scores.setdefault(idx_to_prod[idx], []).append(float(prob))

    recommendations = [
        {"product": product, "prob": round(float(np.mean(values)), 3)}
        for product, values in scores.items()
    ]
    recommendations.sort(key=lambda item: item["prob"], reverse=True)
    return recommendations[:limit]


def build_results(pipeline, model, device):
    loader = app_state["test_loader"]
    pred_logits, true_meds, pred_gaps, true_gaps, cids, target_days = model.predict(loader, device=device, apply_mask=True)
    rows = []
    sequence_lookup = pipeline.test_seq_df.reset_index(drop=True)
    for idx, row in sequence_lookup.iterrows():
        predicted_meds = top_meds_from_logits(pred_logits[idx], pipeline.all_meds)
        recommendations = recommend_products(row["age"], row["disease"], predicted_meds)
        true_products = row["target"].get("products", [])
        rec_names = [item["product"] for item in recommendations]
        intersection = len(set(rec_names).intersection(true_products))
        union = len(set(rec_names).union(true_products)) or 1
        retail_recall = intersection / max(1, len(set(true_products)))
        med_intersection = len(set(predicted_meds).intersection(row["target"]["meds"]))
        med_recall = med_intersection / max(1, len(set(row["target"]["meds"])))
        loc = get_client_location(row["client_id"])
        target_pharmacy_day_val = int(round(float(target_days[idx])))
        predicted_pharmacy_day_val = int(round(float(target_days[idx] - true_gaps[idx].item() + pred_gaps[idx].item())))
        prev_pharmacy_day = row.get("prev_pharmacy_day")
        if prev_pharmacy_day is None:
            prev_pharmacy_day = target_pharmacy_day_val
        predicted_days_later = max(0, predicted_pharmacy_day_val - int(prev_pharmacy_day))
        actual_days_later = max(0, target_pharmacy_day_val - int(prev_pharmacy_day))
        rows.append(
            {
                "client_id": row["client_id"],
                "age": int(row["age"]),
                "disease": row["disease"],
                "target_pharmacy_day": target_pharmacy_day_val,
                "predicted_pharmacy_day": predicted_pharmacy_day_val,
                "predicted_days_later": predicted_days_later,
                "actual_days_later": actual_days_later,
                "actual_gap": int(round(float(true_gaps[idx].item()))),
                "predicted_gap": int(round(float(pred_gaps[idx].item()))),
                "true_meds": "|".join(row["target"]["meds"]),
                "predicted_meds": "|".join(predicted_meds),
                "true_products": "|".join(true_products),
                "recommended_products": "|".join(rec_names),
                "retail_jaccard": round(intersection / union, 3),
                "retail_recall": round(retail_recall, 3),
                "med_recall": round(med_recall, 3),
                "day_delta": int(round(float(target_days[idx] - true_gaps[idx].item() + pred_gaps[idx].item()) - float(target_days[idx]))),
                "store_name": loc["store_name"],
                "store_address": loc["store_address"],
                "store_lat": loc["store_lat"],
                "store_lon": loc["store_lon"],
                "client_lat": loc["client_lat"],
                "client_lon": loc["client_lon"],
            }
        )
    return pd.DataFrame(rows)


def predict_one_step(pipeline, history_window, age, disease, target_day):
    model = app_state["lstm_model"]
    if model is None:
        raise ValueError("Model is not trained.")
    row = pd.DataFrame(
        [
            {
                "client_id": "_future",
                "age": int(age),
                "disease": disease,
                "target_pharmacy_day": int(target_day),
                "history_window": history_window,
                "target": {"M_gap": 0.0, "meds": [], "products": []},
            }
        ]
    )
    dataset = ClinicalDataset(row, pipeline.all_meds, pipeline.disease_to_idx, seq_len=pipeline.seq_len)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_logits, _, pred_gaps, _, _, _ = model.predict(loader, device=device, apply_mask=True)
    predicted_meds = top_meds_from_logits(pred_logits[0], pipeline.all_meds)
    predicted_gap = int(round(max(0.0, float(pred_gaps[0].item()))))
    recommendations = recommend_products(age, disease, predicted_meds)
    return predicted_gap, predicted_meds, recommendations


def med_duration_lookup(pipeline):
    durations = {}
    exploded = pipeline.df[["meds_received", "m_days"]].explode("meds_received")
    for med, group in exploded.groupby("meds_received"):
        durations[med] = int(round(float(group["m_days"].median())))
    return durations


def forecast_client(pipeline, uid, horizon_days):
    patient = pipeline.df[pipeline.df["client_id"] == uid].sort_values("pharmacy_report_day")
    if patient.empty:
        return {"context": None, "forecasts": []}
    duration_map = med_duration_lookup(pipeline)
    age = int(patient["age"].iloc[0])
    disease = patient["disease"].iloc[0]
    visits = patient.tail(pipeline.seq_len).to_dict("records")
    history = [
        {"M_gap": float(row["gap"]), "meds": row["meds_received"], "m_days": int(row["m_days"])}
        for row in visits
    ]
    last_day = int(patient["pharmacy_report_day"].iloc[-1])
    last_meds = patient["meds_received"].iloc[-1]
    last_products = patient["retail_products"].iloc[-1]
    end_day = last_day + int(horizon_days)
    forecasts = []
    current_day = last_day
    last_m_days = int(patient["m_days"].iloc[-1])
    loc = get_client_location(uid)
    while current_day <= end_day and len(forecasts) < 20:
        predicted_gap, predicted_meds, recommendations = predict_one_step(
            pipeline,
            history[-pipeline.seq_len:],
            age,
            disease,
            current_day,
        )
        predicted_day = int(round(current_day + last_m_days + predicted_gap))
        if predicted_day > end_day:
            break
        next_m_days = max([duration_map.get(med, 30) for med in predicted_meds] or [30])
        forecasts.append(
            {
                "client_id": uid,
                "age": age,
                "disease": disease,
                "base_day": current_day,
                "predicted_pharmacy_day": predicted_day,
                "days_later": int(round(predicted_day - last_day)),
                "predicted_gap": int(round(predicted_gap)),
                "predicted_meds": "|".join(predicted_meds),
                "recommended_products": "|".join(item["product"] for item in recommendations),
                "store_name": loc["store_name"],
                "store_address": loc["store_address"],
                "store_lat": loc["store_lat"],
                "store_lon": loc["store_lon"],
                "client_lat": loc["client_lat"],
                "client_lon": loc["client_lon"],
            }
        )
        history.append({"M_gap": predicted_gap, "meds": predicted_meds, "m_days": next_m_days})
        current_day = int(round(predicted_day))
        last_m_days = next_m_days
    return {
        "context": {
            "client_id": uid,
            "age": age,
            "disease": disease,
            "last_pharmacy_day": last_day,
            "last_m_days": int(patient["m_days"].iloc[-1]),
            "coverage_end_day": last_day + int(patient["m_days"].iloc[-1]),
            "last_meds": last_meds,
            "last_products": last_products,
            "store_name": loc["store_name"],
            "store_address": loc["store_address"],
            "store_lat": loc["store_lat"],
            "store_lon": loc["store_lon"],
            "client_lat": loc["client_lat"],
            "client_lon": loc["client_lon"],
        },
        "forecasts": forecasts,
    }


def metrics_from_results(results_df):
    day_mae = float(np.abs(results_df["predicted_gap"] - results_df["actual_gap"]).mean())
    med_scores = []
    retail_scores = []
    for _, row in results_df.iterrows():
        med_scores.append(float(row.get("med_recall", 0.0)))
        retail_scores.append(float(row.get("retail_recall", 0.0)))
    return {
        "day_mae": round(day_mae, 2),
        "med_recall": round(float(np.mean(med_scores)), 3),
        "retail_recall": round(float(np.mean(retail_scores)), 3),
        "evaluated_rows": int(len(results_df)),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload_data", methods=["POST"])
def upload_data():
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded."}), 400
    file = request.files["file"]
    mode = request.form.get("mode", "overwrite")
    seq_len = int(request.form.get("seq_len", 5))
    split_ratio = float(request.form.get("split_ratio", 0.8))
    
    try:
        new_df = pd.read_csv(file)
        missing = sorted(REQUIRED_COLUMNS - set(new_df.columns))
        if missing:
            return jsonify({"status": "error", "message": f"Missing columns: {', '.join(missing)}"}), 400
            
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        if mode == "overwrite":
            cursor.execute("DELETE FROM clinical_records")
            
        for _, row in new_df.iterrows():
            cursor.execute("""
            INSERT INTO clinical_records (client_id, age, disease, doctor_visit_day, pharmacy_report_day, m_days, meds_received, retail_products)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(row["client_id"]),
                int(row["age"]),
                str(row["disease"]),
                int(row["doctor_visit_day"]),
                int(row["pharmacy_report_day"]),
                int(row["m_days"]),
                str(row["meds_received"]),
                str(row["retail_products"])
            ))
        conn.commit()
        
        df = pd.read_sql_query("SELECT * FROM clinical_records", conn)
        conn.close()
        
        numeric_cols = ["age", "doctor_visit_day", "pharmacy_report_day", "m_days"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            
        pipeline = UploadedPipeline(df, split_ratio=split_ratio, seq_len=seq_len)
        if pipeline.train_seq_df.empty or pipeline.test_seq_df.empty:
            return jsonify({"status": "error", "message": "Uploaded data must contain both train and test time windows."}), 400
            
        reset_training_state()
        app_state["pipeline"] = pipeline
        app_state["train_loader"] = DataLoader(
            ClinicalDataset(pipeline.train_seq_df, pipeline.all_meds, pipeline.disease_to_idx, seq_len=seq_len),
            batch_size=32,
            shuffle=True,
        )
        app_state["test_loader"] = DataLoader(
            ClinicalDataset(pipeline.test_seq_df, pipeline.all_meds, pipeline.disease_to_idx, seq_len=seq_len),
            batch_size=32,
            shuffle=False,
        )
        summary = build_summary(pipeline)
        return jsonify(
            {
                "status": "success",
                "summary": summary,
                "summary_plot": plot_summary(summary),
                "timeline_preview": plot_model_view_preview(pipeline),
            }
        )
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/clear_upload", methods=["POST"])
def clear_upload():
    app_state["pipeline"] = None
    reset_training_state()
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM clinical_records")
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error clearing db upload: {e}")
    return jsonify({"status": "success"})


@app.route("/api/train", methods=["POST"])
def train_models():
    pipeline = app_state["pipeline"]
    if pipeline is None:
        return jsonify({"status": "error", "message": "Upload fake_data.csv before training."}), 400

    data = request.json or {}
    epochs = int(data.get("epochs", 5))
    retail_epochs = int(data.get("retail_epochs", 8))
    diff_epochs = int(data.get("diff_epochs", 8))
    diff_steps = int(data.get("diff_steps", 30))
    lr = float(data.get("lr", 0.005))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logs = ["Building LSTM visit model."]

    lstm_model = BaselineLSTM(
        len(pipeline.all_meds) + 1,
        len(pipeline.disease_names),
        len(pipeline.all_meds),
        pipeline.disease_mask,
    )
    lstm_losses = lstm_model.fit(app_state["train_loader"], epochs=epochs, lr=lr, device=device)
    for epoch_idx, loss_val in enumerate(lstm_losses):
        logs.append(f"LSTM epoch {epoch_idx + 1}/{epochs}: loss={loss_val:.4f}")
    logs.append(f"LSTM complete: {epochs} epochs.")

    retail_model, retail_logs = train_retail_nn(pipeline, epochs=retail_epochs)
    logs.extend(retail_logs)
    
    logs.append("Training retail diffusion model.")
    diffusion_model, encoder, diffusion_losses = train_retail_diffusion(pipeline, epochs=diff_epochs, steps=diff_steps)
    for epoch_idx, loss_val in enumerate(diffusion_losses):
        logs.append(f"Diffusion epoch {epoch_idx + 1}/{diff_epochs}: loss={loss_val:.4f}")
    logs.append(f"Retail diffusion complete: {diff_epochs} epochs, {diff_steps} steps.")

    app_state["lstm_model"] = lstm_model
    app_state["retail_model"] = retail_model
    app_state["retail_diffusion_model"] = diffusion_model
    app_state["retail_disease_encoder"] = encoder
    app_state["retail_med_to_idx"] = pipeline.med_to_idx
    app_state["retail_prod_to_idx"] = pipeline.prod_to_idx
    app_state["retail_products"] = pipeline.products
    app_state["test_uids"] = pipeline.test_seq_df["client_id"].unique().tolist()
    app_state["results_df"] = build_results(pipeline, lstm_model, device)
    app_state["model_analysis_plot"] = plot_medication_analysis(
        pipeline,
        app_state["results_df"],
        app_state["test_uids"][0] if app_state["test_uids"] else None,
    )
    app_state["training_log"] = logs

    # Save model checkpoint
    import datetime
    metrics = metrics_from_results(app_state["results_df"])
    model_id = f"model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        save_model_checkpoint(
            model_id=model_id,
            lstm=lstm_model,
            retail=retail_model,
            diffusion=diffusion_model,
            encoder=encoder,
            pipeline=pipeline,
            logs=logs,
            metrics=metrics,
            epochs=epochs,
            retail_epochs=retail_epochs,
            diff_epochs=diff_epochs,
            lr=lr
        )
        app_state["active_model_id"] = model_id
    except Exception as e:
        print(f"Error saving trained model checkpoint: {e}")
        app_state["active_model_id"] = ""

    return jsonify(
        {
            "status": "success",
            "logs": logs,
            "metrics": metrics,
            "test_uids": app_state["test_uids"],
            "model_analysis_plot": app_state["model_analysis_plot"],
            "active_model_id": app_state.get("active_model_id", "")
        }
    )


@app.route("/api/results")
def results():
    if app_state["results_df"] is None:
        return jsonify({"status": "error", "message": "Train models before viewing results."}), 400
    rows = app_state["results_df"].head(200).to_dict("records")
    return jsonify({"status": "success", "metrics": metrics_from_results(app_state["results_df"]), "rows": rows})


@app.route("/api/patient/<uid>")
def patient(uid):
    pipeline = app_state["pipeline"]
    results_df = app_state["results_df"]
    if pipeline is None or results_df is None:
        return jsonify({"status": "error", "message": "Upload data and train models first."}), 400

    p_df = pipeline.df[pipeline.df["client_id"] == uid].sort_values("pharmacy_report_day")
    p_results = results_df[results_df["client_id"] == uid].sort_values("target_pharmacy_day")
    if p_df.empty or p_results.empty:
        return jsonify({"status": "error", "message": "Patient not found in test window."}), 404

    timeline = [
        {
            "doctor_visit_day": int(row["doctor_visit_day"]),
            "pharmacy_report_day": int(row["pharmacy_report_day"]),
            "m_days": int(row["m_days"]),
            "meds": row["meds_received"],
            "products": row["retail_products"],
        }
        for _, row in p_df.iterrows()
    ]
    predictions = p_results.to_dict("records")
    latest = p_df.iloc[-1]
    current_meds = latest["meds_received"]
    return jsonify(
        {
            "status": "success",
            "uid": uid,
            "age": int(p_df["age"].iloc[0]),
            "disease": p_df["disease"].iloc[0],
            "current_meds": current_meds,
            "history_cutoff": int(pipeline.split_day),
            "timeline": timeline,
            "predictions": predictions,
        }
    )


@app.route("/api/model_analysis_plot/<uid>")
def model_analysis_plot(uid):
    pipeline = app_state["pipeline"]
    results_df = app_state["results_df"]
    if pipeline is None or results_df is None:
        return jsonify({"status": "error", "message": "Train models first."}), 400
    return jsonify({"status": "success", "plot": plot_medication_analysis(pipeline, results_df, uid)})


@app.route("/api/future_forecast")
def future_forecast():
    pipeline = app_state["pipeline"]
    if pipeline is None or app_state["lstm_model"] is None:
        return jsonify({"status": "error", "message": "Upload data and train models first."}), 400
    horizon_days = int(request.args.get("days", 180))
    uid = request.args.get("uid", "all")
    if uid == "all":
        uids = pipeline.df["client_id"].drop_duplicates().tolist()
    else:
        uids = [uid]
    forecasts = []
    contexts = []
    for client_id in uids:
        forecast = forecast_client(pipeline, client_id, horizon_days)
        if forecast["context"] is not None:
            contexts.append(forecast["context"])
            forecasts.extend(forecast["forecasts"])
    return jsonify({"status": "success", "days": horizon_days, "contexts": contexts, "forecasts": forecasts})


@app.route("/api/download_results")
def download_results():
    if app_state["results_df"] is None:
        return jsonify({"status": "error", "message": "No results available."}), 400
    
    lang = request.args.get("lang", "zh")
    df_to_export = app_state["results_df"].copy()
    if lang == "en":
        from fake_locations import CONVENIENCE_STORES
        name_map = {store["name"]: store["name_en"] for store in CONVENIENCE_STORES}
        addr_map = {store["address"]: store["address_en"] for store in CONVENIENCE_STORES}
        df_to_export["store_name"] = df_to_export["store_name"].map(name_map).fillna(df_to_export["store_name"])
        df_to_export["store_address"] = df_to_export["store_address"].map(addr_map).fillna(df_to_export["store_address"])
    elif lang == "zh":
        df_to_export = df_to_export.rename(columns={
            "client_id": "使用者ID",
            "age": "年齡",
            "disease": "疾病",
            "target_pharmacy_day": "實際時間",
            "predicted_pharmacy_day": "預測時間",
            "predicted_days_later": "預測幾天後",
            "actual_days_later": "實際幾天後",
            "actual_gap": "實際間隔",
            "predicted_gap": "預測間隔",
            "true_meds": "實際藥品",
            "predicted_meds": "預測藥品",
            "true_products": "實際商品",
            "recommended_products": "推薦商品",
            "retail_jaccard": "商品Jaccard相似度",
            "retail_recall": "商品Recall",
            "med_recall": "藥品Recall",
            "day_delta": "時間差",
            "store_name": "分配超商",
            "store_address": "超商地址",
            "store_lat": "超商緯度",
            "store_lon": "超商經度",
            "client_lat": "使用者緯度",
            "client_lon": "使用者經度",
        })

    csv_data = df_to_export.to_csv(index=False)
    mem = io.BytesIO()
    mem.write(csv_data.encode("utf-8-sig"))
    mem.seek(0)
    return send_file(
        mem,
        mimetype="text/csv",
        as_attachment=True,
        download_name="prediction_results.csv"
    )


@app.route("/api/download_future_forecast")
def download_future_forecast():
    pipeline = app_state["pipeline"]
    if pipeline is None or app_state["lstm_model"] is None:
        return jsonify({"status": "error", "message": "Upload data and train models first."}), 400
    horizon_days = int(request.args.get("days", 180))
    uid = request.args.get("uid", "all")
    lang = request.args.get("lang", "zh")
    if uid == "all":
        uids = pipeline.df["client_id"].drop_duplicates().tolist()
    else:
        uids = [uid]
    forecasts = []
    for client_id in uids:
        forecast = forecast_client(pipeline, client_id, horizon_days)
        if forecast["context"] is not None:
            forecasts.extend(forecast["forecasts"])
    if not forecasts:
        return jsonify({"status": "error", "message": "No forecasts found."}), 400

    # Calculate model mean absolute error (MAE)
    results_df = app_state.get("results_df")
    day_mae = 4.5
    if results_df is not None:
        try:
            import numpy as np
            day_mae = float(np.abs(results_df["predicted_gap"] - results_df["actual_gap"]).mean())
        except Exception:
            pass

    # Parse simulation base date
    import datetime
    base_date_str = request.args.get("base_date", "2026-07-06")
    try:
        base_date = datetime.datetime.strptime(base_date_str, "%Y-%m-%d")
    except Exception:
        try:
            base_date = datetime.datetime.strptime(base_date_str, "%m/%d/%Y")
        except Exception:
            base_date = datetime.datetime.strptime("2026-07-06", "%Y-%m-%d")

    # Add localized fields
    for row in forecasts:
        days_later = row.get("days_later", 0)
        pred_date = base_date + datetime.timedelta(days=int(round(days_later)))
        row["predicted_date"] = pred_date.strftime("%Y-%m-%d")
        row["expected_deviation"] = f"± {day_mae:.1f}d"

    df_forecast = pd.DataFrame(forecasts)
    if lang == "en":
        from fake_locations import CONVENIENCE_STORES
        name_map = {store["name"]: store["name_en"] for store in CONVENIENCE_STORES}
        addr_map = {store["address"]: store["address_en"] for store in CONVENIENCE_STORES}
        df_forecast["store_name"] = df_forecast["store_name"].map(name_map).fillna(df_forecast["store_name"])
        df_forecast["store_address"] = df_forecast["store_address"].map(addr_map).fillna(df_forecast["store_address"])
        df_forecast = df_forecast.rename(columns={
            "predicted_date": "Predicted Date",
            "expected_deviation": "Expected Deviation",
        })
        first_cols = ["client_id", "age", "disease", "Predicted Date", "Expected Deviation"]
        remaining_cols = [c for c in df_forecast.columns if c not in first_cols]
        df_forecast = df_forecast[[c for c in first_cols if c in df_forecast.columns] + remaining_cols]
    elif lang == "zh":
        df_forecast = df_forecast.rename(columns={
            "client_id": "使用者ID",
            "age": "年齡",
            "disease": "疾病",
            "base_day": "補藥基準點",
            "predicted_pharmacy_day": "預測時間",
            "days_later": "幾天後",
            "predicted_gap": "預測間隔",
            "predicted_meds": "預測藥品",
            "recommended_products": "推薦商品",
            "predicted_date": "預估日期",
            "expected_deviation": "預期偏差",
            "store_name": "分配超商",
            "store_address": "超商地址",
            "store_lat": "超商緯度",
            "store_lon": "超商經度",
            "client_lat": "使用者緯度",
            "client_lon": "使用者經度",
        })
        first_cols = ["使用者ID", "年齡", "疾病", "預估日期", "預期偏差"]
        remaining_cols = [c for c in df_forecast.columns if c not in first_cols]
        df_forecast = df_forecast[[c for c in first_cols if c in df_forecast.columns] + remaining_cols]

    csv_data = df_forecast.to_csv(index=False)
    mem = io.BytesIO()
    mem.write(csv_data.encode("utf-8-sig"))
    mem.seek(0)
    filename = f"future_forecast_{uid}_{horizon_days}d.csv"
    return send_file(
        mem,
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename
    )


def fetch_stores_from_api(limit):
    import time
    queries = [
        {"q": "7-11 Tainan", "brand": "7-11"},
        {"q": "FamilyMart Tainan", "brand": "FamilyMart"},
        {"q": "7-11 Kaohsiung", "brand": "7-11"},
        {"q": "FamilyMart Kaohsiung", "brand": "FamilyMart"},
        {"q": "7-11 Pingtung", "brand": "7-11"},
        {"q": "FamilyMart Pingtung", "brand": "FamilyMart"}
    ]
    
    sub_limit = math.ceil(limit / len(queries))
    url = "https://nominatim.openstreetmap.org/search"
    headers = {
        "User-Agent": "PharmacySimulatorApp/1.0 (contact: support@pharmacy.example.com)"
    }
    
    fetched_stores = {}
    
    for q_info in queries:
        q_str = q_info["q"]
        brand = q_info["brand"]
        
        # 1. Fetch Chinese
        zh_headers = headers.copy()
        zh_headers["Accept-Language"] = "zh-TW"
        zh_params = {
            "q": q_str,
            "format": "json",
            "limit": sub_limit,
            "addressdetails": 1
        }
        
        zh_results = []
        try:
            r = requests.get(url, params=zh_params, headers=zh_headers, timeout=15)
            time.sleep(1.0)  # Respect Nominatim rate limit (max 1 req/sec)
            if r.status_code == 200:
                zh_results = r.json()
            else:
                print(f"Nominatim returned {r.status_code} for Chinese {q_str}")
        except Exception as e:
            print(f"Error fetching Chinese for {q_str}: {e}")
            
        # 2. Fetch English
        en_headers = headers.copy()
        en_headers["Accept-Language"] = "en"
        en_params = {
            "q": q_str,
            "format": "json",
            "limit": sub_limit,
            "addressdetails": 1
        }
        
        en_results = {}
        try:
            r = requests.get(url, params=en_params, headers=en_headers, timeout=15)
            time.sleep(1.0)  # Respect Nominatim rate limit (max 1 req/sec)
            if r.status_code == 200:
                for item in r.json():
                    osm_id = f"{item.get('osm_type')}_{item.get('osm_id')}"
                    en_results[osm_id] = item
            else:
                print(f"Nominatim returned {r.status_code} for English {q_str}")
        except Exception as e:
            print(f"Error fetching English for {q_str}: {e}")
            
        # Process and combine
        for item in zh_results:
            osm_type = item.get("osm_type")
            osm_val = item.get("osm_id")
            if not osm_val:
                continue
            osm_id = f"{osm_type}_{osm_val}"
            
            lat = item.get("lat")
            lon = item.get("lon")
            if not lat or not lon:
                continue
            lat = float(lat)
            lon = float(lon)
            
            # Chinese address construction
            zh_addr = item.get("address", {})
            city = zh_addr.get("city") or zh_addr.get("county") or ""
            town = zh_addr.get("town") or ""
            suburb = zh_addr.get("suburb") or ""
            village = zh_addr.get("village") or zh_addr.get("neighbourhood") or ""
            road = zh_addr.get("road") or ""
            housenumber = zh_addr.get("house_number") or ""
            
            parts = []
            if city:
                parts.append(city)
            if town and town not in parts:
                parts.append(town)
            if suburb and suburb not in parts and suburb not in town:
                parts.append(suburb)
            if village and village not in parts and village not in suburb and village not in town:
                parts.append(village)
            if road:
                parts.append(road)
            if housenumber:
                parts.append(housenumber + "號" if not housenumber.endswith("號") else housenumber)
                
            address = "".join(parts)
            if not address:
                address = item.get("display_name", "")
                
            # Nice descriptive Chinese name (e.g. 7-11 大學門市)
            ref_name = road.replace("路", "").replace("街", "").replace("段", "")
            if not ref_name:
                ref_name = suburb or town or str(osm_val)[:5]
            suffix = "門市" if brand == "7-11" else "店"
            name = f"{brand} {ref_name}{suffix}"
                
            # English address and name construction
            en_item = en_results.get(osm_id)
            if en_item:
                en_addr = en_item.get("address", {})
                en_city = en_addr.get("city") or en_addr.get("county") or ""
                en_town = en_addr.get("town") or ""
                en_suburb = en_addr.get("suburb") or ""
                en_village = en_addr.get("village") or en_addr.get("neighbourhood") or ""
                en_road = en_addr.get("road") or ""
                en_housenumber = en_addr.get("house_number") or ""
                
                en_parts = []
                if en_housenumber:
                    en_parts.append(en_housenumber)
                if en_road:
                    en_parts.append(en_road)
                if en_village and en_village not in en_parts:
                    en_parts.append(en_village)
                if en_suburb and en_suburb not in en_parts and en_suburb not in en_town:
                    en_parts.append(en_suburb)
                if en_town and en_town not in en_parts:
                    en_parts.append(en_town)
                if en_city and en_city not in en_parts:
                    en_parts.append(en_city)
                
                address_en = ", ".join(en_parts)
                if not address_en:
                    address_en = en_item.get("display_name", "")
                
                en_ref = en_road.replace("Road", "").replace("Street", "").replace("Sec.", "").strip()
                if not en_ref:
                    en_ref = en_suburb or en_town or str(osm_val)[:5]
                name_en = f"{brand} {en_ref} Store"
            else:
                address_en = f"Convenience Store, Taiwan ({lat:.4f}, {lon:.4f})"
                name_en = f"{brand} Store {osm_val}"
                
            fetched_stores[osm_id] = {
                "osm_id": osm_id,
                "name": name,
                "address": address,
                "name_en": name_en,
                "address_en": address_en,
                "lat": lat,
                "lon": lon
            }
            
            if len(fetched_stores) >= limit:
                break
        
        if len(fetched_stores) >= limit:
            break
            
    return list(fetched_stores.values())[:limit]


@app.route("/api/convenience_stores")
def get_convenience_stores():
    from fake_locations import CONVENIENCE_STORES
    db_count = 0
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM convenience_stores")
            db_count = cursor.fetchone()[0]
            conn.close()
        except Exception:
            pass
            
    return jsonify({
        "status": "success",
        "stores": CONVENIENCE_STORES,
        "db_count": db_count,
        "is_using_db": db_count > 0
    })


@app.route("/api/fetch_stores", methods=["POST"])
def fetch_stores():
    data = request.get_json() or {}
    limit = data.get("limit")
    if not limit:
        return jsonify({"status": "error", "message": "Limit parameter is required."}), 400
    try:
        limit = int(limit)
        if limit < 10 or limit > 1000:
            return jsonify({"status": "error", "message": "Limit must be between 10 and 1000."}), 400
    except ValueError:
        return jsonify({"status": "error", "message": "Limit must be an integer."}), 400
        
    stores = fetch_stores_from_api(limit)
    if not stores:
        return jsonify({"status": "error", "message": "Failed to fetch stores from API. Please try again."}), 500
        
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM convenience_stores")
        for s in stores:
            cursor.execute("""
            INSERT OR REPLACE INTO convenience_stores (osm_id, name, address, name_en, address_en, lat, lon)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (s["osm_id"], s["name"], s["address"], s["name_en"], s["address_en"], s["lat"], s["lon"]))
        conn.commit()
        conn.close()
        
        # Reload fake_locations list
        from fake_locations import reload_stores
        reload_stores()
        
        return jsonify({
            "status": "success",
            "message": f"Successfully fetched and saved {len(stores)} stores to local database.",
            "count": len(stores)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"Database error: {str(e)}"}), 500



MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

def save_model_checkpoint(model_id, lstm, retail, diffusion, encoder, pipeline, logs, metrics, epochs, retail_epochs, diff_epochs, lr):
    import datetime
    import json
    os.makedirs(MODELS_DIR, exist_ok=True)
    dir_path = os.path.join(MODELS_DIR, model_id)
    os.makedirs(dir_path, exist_ok=True)
    
    # Save PyTorch state dicts
    torch.save(lstm.state_dict(), os.path.join(dir_path, "lstm.pt"))
    torch.save(retail.state_dict(), os.path.join(dir_path, "retail.pt"))
    torch.save(diffusion.state_dict(), os.path.join(dir_path, "diffusion.pt"))
    
    # Save meta json
    meta = {
        "model_id": model_id,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "db_name": app_state.get("active_db", "default"),
        "epochs": epochs,
        "retail_epochs": retail_epochs,
        "diff_epochs": diff_epochs,
        "lr": lr,
        "metrics": metrics,
        "logs": logs,
        "disease_names": pipeline.disease_names,
        "all_meds": pipeline.all_meds,
        "products": pipeline.products,
        "disease_to_idx": pipeline.disease_to_idx,
        "med_to_idx": pipeline.med_to_idx,
        "prod_to_idx": pipeline.prod_to_idx,
        "disease_mask": pipeline.disease_mask.tolist() if hasattr(pipeline.disease_mask, 'tolist') else pipeline.disease_mask,
        "encoder_classes": encoder.classes_.tolist() if hasattr(encoder, 'classes_') else []
    }
    with open(os.path.join(dir_path, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def reload_pipeline_from_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM clinical_records", conn)
        conn.close()
        if not df.empty:
            numeric_cols = ["age", "doctor_visit_day", "pharmacy_report_day", "m_days"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            seq_len = 5
            split_ratio = 0.8
            if app_state["pipeline"] is not None:
                seq_len = app_state["pipeline"].seq_len
                split_ratio = app_state["pipeline"].split_ratio
            app_state["pipeline"] = UploadedPipeline(df, split_ratio=split_ratio, seq_len=seq_len)
            
            # Recreate data loaders
            app_state["train_loader"] = DataLoader(
                ClinicalDataset(app_state["pipeline"].train_seq_df, app_state["pipeline"].all_meds, app_state["pipeline"].disease_to_idx, seq_len=seq_len),
                batch_size=32, shuffle=True
            )
            app_state["test_loader"] = DataLoader(
                ClinicalDataset(app_state["pipeline"].test_seq_df, app_state["pipeline"].all_meds, app_state["pipeline"].disease_to_idx, seq_len=seq_len),
                batch_size=32, shuffle=False
            )
        else:
            app_state["pipeline"] = None
            reset_training_state()
    except Exception as e:
        print(f"Failed to reload pipeline from database: {e}")


@app.route("/api/records")
def get_records():
    page = int(request.args.get("page", 1))
    limit = int(request.args.get("limit", 50))
    search = request.args.get("search", "").strip()
    
    offset = (page - 1) * limit
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        if search:
            query = "SELECT id, client_id, age, disease, doctor_visit_day, pharmacy_report_day, m_days, meds_received, retail_products FROM clinical_records WHERE client_id LIKE ? OR disease LIKE ? ORDER BY id DESC LIMIT ? OFFSET ?"
            params = (f"%{search}%", f"%{search}%", limit, offset)
            
            cursor.execute("SELECT COUNT(*) FROM clinical_records WHERE client_id LIKE ? OR disease LIKE ?", (f"%{search}%", f"%{search}%"))
            total = cursor.fetchone()[0]
        else:
            query = "SELECT id, client_id, age, disease, doctor_visit_day, pharmacy_report_day, m_days, meds_received, retail_products FROM clinical_records ORDER BY id DESC LIMIT ? OFFSET ?"
            params = (limit, offset)
            
            cursor.execute("SELECT COUNT(*) FROM clinical_records")
            total = cursor.fetchone()[0]
            
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        records = []
        for r in rows:
            client_id = r[1]
            loc = get_client_location(client_id)
            records.append({
                "id": r[0],
                "client_id": client_id,
                "age": r[2],
                "disease": r[3],
                "doctor_visit_day": r[4],
                "pharmacy_report_day": r[5],
                "m_days": r[6],
                "meds_received": r[7],
                "retail_products": r[8],
                "store_name": loc["store_name"],
                "store_address": loc["store_address"],
                "store_lat": loc["store_lat"],
                "store_lon": loc["store_lon"],
                "client_lat": loc["client_lat"],
                "client_lon": loc["client_lon"]
            })
            
        return jsonify({
            "status": "success",
            "records": records,
            "total": total,
            "page": page,
            "pages": math.ceil(total / limit)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/records/update", methods=["POST"])
def update_record():
    data = request.json or {}
    record_id = data.get("id")
    if not record_id:
        return jsonify({"status": "error", "message": "Record ID is required."}), 400
        
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
        UPDATE clinical_records
        SET age = ?, disease = ?, doctor_visit_day = ?, pharmacy_report_day = ?, m_days = ?, meds_received = ?, retail_products = ?
        WHERE id = ?
        """, (
            int(data["age"]),
            str(data["disease"]),
            int(data["doctor_visit_day"]),
            int(data["pharmacy_report_day"]),
            int(data["m_days"]),
            str(data["meds_received"]),
            str(data["retail_products"]),
            record_id
        ))
        conn.commit()
        conn.close()
        
        reload_pipeline_from_db()
        return jsonify({"status": "success", "message": "Record updated successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/records/delete", methods=["POST"])
def delete_record():
    data = request.json or {}
    record_id = data.get("id")
    if not record_id:
        return jsonify({"status": "error", "message": "Record ID is required."}), 400
        
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM clinical_records WHERE id = ?", (record_id,))
        conn.commit()
        conn.close()
        
        reload_pipeline_from_db()
        return jsonify({"status": "success", "message": "Record deleted successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/records/add", methods=["POST"])
def add_record():
    data = request.json or {}
    client_id = data.get("client_id")
    if not client_id:
        return jsonify({"status": "error", "message": "client_id is required."}), 400
        
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO clinical_records (client_id, age, disease, doctor_visit_day, pharmacy_report_day, m_days, meds_received, retail_products)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(client_id),
            int(data.get("age", 50)),
            str(data.get("disease", "Hypertension")),
            int(data.get("doctor_visit_day", 1)),
            int(data.get("pharmacy_report_day", 30)),
            int(data.get("m_days", 30)),
            str(data.get("meds_received", "")),
            str(data.get("retail_products", ""))
        ))
        conn.commit()
        conn.close()
        
        reload_pipeline_from_db()
        return jsonify({"status": "success", "message": "Record added successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/stores/update", methods=["POST"])
def update_store():
    data = request.json or {}
    osm_id = data.get("osm_id")
    if not osm_id:
        return jsonify({"status": "error", "message": "Store identifier (osm_id) is required."}), 400
        
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
        UPDATE convenience_stores
        SET name = ?, address = ?, name_en = ?, address_en = ?, lat = ?, lon = ?
        WHERE osm_id = ?
        """, (
            str(data["name"]),
            str(data["address"]),
            str(data["name_en"]),
            str(data["address_en"]),
            float(data["lat"]),
            float(data["lon"]),
            osm_id
        ))
        conn.commit()
        conn.close()
        
        from fake_locations import reload_stores
        reload_stores()
        
        return jsonify({"status": "success", "message": "Store updated successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/stores/delete", methods=["POST"])
def delete_store():
    data = request.json or {}
    osm_id = data.get("osm_id")
    if not osm_id:
        return jsonify({"status": "error", "message": "Store identifier (osm_id) is required."}), 400
        
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM convenience_stores WHERE osm_id = ?", (osm_id,))
        conn.commit()
        conn.close()
        
        from fake_locations import reload_stores
        reload_stores()
        
        return jsonify({"status": "success", "message": "Store deleted successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/stores/add", methods=["POST"])
def add_store():
    data = request.json or {}
    name = data.get("name")
    if not name:
        return jsonify({"status": "error", "message": "Store name is required."}), 400
        
    import random
    osm_id = f"custom_{random.randint(100000, 999999)}"
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO convenience_stores (osm_id, name, address, name_en, address_en, lat, lon)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            osm_id,
            str(name),
            str(data.get("address", "")),
            str(data.get("name_en", name)),
            str(data.get("address_en", "")),
            float(data.get("lat", 22.9)),
            float(data.get("lon", 120.2))
        ))
        conn.commit()
        conn.close()
        
        from fake_locations import reload_stores
        reload_stores()
        
        return jsonify({"status": "success", "message": "Store added successfully.", "osm_id": osm_id})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/stores/<osm_id>/clients")
def get_store_clients(osm_id):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM convenience_stores WHERE osm_id = ?", (osm_id,))
        store_row = cursor.fetchone()
        conn.close()
        if not store_row:
            return jsonify({"status": "error", "message": "Store not found."}), 404
        store_name = store_row[0]
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT client_id FROM clinical_records")
        client_rows = cursor.fetchall()
        conn.close()
        
        store_clients = []
        for r in client_rows:
            uid = r[0]
            loc = get_client_location(uid)
            if loc["store_name"] == store_name:
                store_clients.append({
                    "client_id": uid,
                    "client_lat": loc["client_lat"],
                    "client_lon": loc["client_lon"]
                })
        return jsonify({"status": "success", "clients": store_clients})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/models")
def list_models():
    import json
    os.makedirs(MODELS_DIR, exist_ok=True)
    models = []
    active_id = app_state.get("active_model_id", "")
    active_db = app_state.get("active_db", "default")
    
    for folder in sorted(os.listdir(MODELS_DIR)):
        folder_path = os.path.join(MODELS_DIR, folder)
        if os.path.isdir(folder_path):
            meta_path = os.path.join(folder_path, "meta.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    # Only return models associated with the currently active database
                    linked_db = meta.get("db_name", "default")
                    if linked_db == active_db:
                        meta["active"] = (meta["model_id"] == active_id)
                        models.append(meta)
                except Exception as e:
                    print(f"Error reading model meta {folder}: {e}")
    models = sorted(models, key=lambda m: m.get("timestamp", ""), reverse=True)
    return jsonify({"status": "success", "models": models, "active_model_id": active_id})



def select_database(db_name):
    app_state["active_db"] = db_name
    global DB_PATH
    DB_PATH = get_db_path()
    fake_locations.set_db_path(DB_PATH)
    fake_locations.reload_stores()
    init_db()
    reload_pipeline_from_db()
    
    # Reset active model state in memory when changing active databases
    app_state["lstm_model"] = None
    app_state["retail_model"] = None
    app_state["retail_diffusion_model"] = None
    app_state["retail_disease_encoder"] = None
    app_state["active_model_id"] = ""


def check_mapping_status():
    try:
        active_db = app_state.get("active_db")
        if not active_db:
            return "unassigned"
            
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if convenience_stores has rows
        cursor.execute("SELECT COUNT(*) FROM convenience_stores")
        stores_count = cursor.fetchone()[0]
        
        # Check if clinical_records has rows
        cursor.execute("SELECT COUNT(*) FROM clinical_records")
        records_count = cursor.fetchone()[0]
        
        if records_count == 0:
            conn.close()
            return "unassigned"
            
        # Check if any row has NULL assigned_store_id
        cursor.execute("SELECT COUNT(*) FROM clinical_records WHERE assigned_store_id IS NULL")
        unassigned_count = cursor.fetchone()[0]
        conn.close()
        
        if unassigned_count > 0:
            return "unassigned"
        else:
            return "assigned"
    except Exception as e:
        print(f"Error checking mapping status: {e}")
        return "unassigned"


@app.route("/api/db_status")
def get_db_status():
    try:
        active_db = app_state.get("active_db")
        if not active_db:
            return jsonify({
                "status": "success",
                "active_db": None,
                "stores_count": 0,
                "records_count": 0,
                "mapping_status": "unassigned",
                "active_model_id": ""
            })
            
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM convenience_stores")
        stores_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM clinical_records")
        records_count = cursor.fetchone()[0]
        conn.close()
        
        mapping = check_mapping_status()
        return jsonify({
            "status": "success",
            "active_db": active_db,
            "stores_count": stores_count,
            "records_count": records_count,
            "mapping_status": mapping,
            "active_model_id": app_state.get("active_model_id", "")
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/records/assign_stores", methods=["POST"])
def assign_stores():
    try:
        # Load convenience stores list
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT osm_id, lat, lon FROM convenience_stores")
        stores = cursor.fetchall()
        
        if not stores:
            conn.close()
            return jsonify({
                "status": "error",
                "message": "資料庫中沒有任何超商資料。請先到「超商資料」分頁獲取或新增超商。"
            }), 400
            
        # Fetch unique client_ids in clinical_records
        cursor.execute("SELECT DISTINCT client_id FROM clinical_records")
        clients = [r[0] for r in cursor.fetchall()]
        
        if not clients:
            conn.close()
            return jsonify({
                "status": "error",
                "message": "資料庫中沒有任何病患紀錄。請先上傳 CSV 資料。"
            }), 400
            
        # For each client, randomly assign a store and generate client coordinates around it
        import random
        for client_id in clients:
            store_osm_id, s_lat, s_lon = random.choice(stores)
            
            # Standard deviation of ~0.005 degrees (approx 500 meters)
            std_dev = 0.005
            client_lat = random.gauss(s_lat, std_dev)
            client_lon = random.gauss(s_lon, std_dev)
            
            cursor.execute("""
            UPDATE clinical_records
            SET assigned_store_id = ?, client_lat = ?, client_lon = ?
            WHERE client_id = ?
            """, (store_osm_id, client_lat, client_lon, client_id))
            
        conn.commit()
        conn.close()
        
        # Reload pipeline to sync changes
        reload_pipeline_from_db()
        
        return jsonify({
            "status": "success",
            "message": f"成功隨機分配 {len(clients)} 位病患至超商門市並產生座標。"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/databases")
def list_databases():
    os.makedirs(DATABASES_DIR, exist_ok=True)
    db_names = [f[:-3] for f in os.listdir(DATABASES_DIR) if f.endswith(".db")]
    
    active_db = app_state.get("active_db")
    if not db_names:
        active_db = None
        app_state["active_db"] = None
    elif active_db not in db_names:
        # Fall back to first available database
        active_db = db_names[0]
        select_database(active_db)
        
    return jsonify({
        "status": "success",
        "databases": sorted(db_names),
        "active": active_db
    })


@app.route("/api/databases/select", methods=["POST"])
def select_db_route():
    data = request.json or {}
    name = data.get("name", "default").strip()
    if not name:
        return jsonify({"status": "error", "message": "Database name is required."}), 400
        
    try:
        select_database(name)
        mapping = check_mapping_status()
        return jsonify({
            "status": "success",
            "message": f"Database '{name}' is selected.",
            "active": name,
            "mapping_status": mapping
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/databases/create", methods=["POST"])
def create_db_route():
    data = request.json or {}
    raw_name = data.get("name", "").strip()
    if not raw_name:
        return jsonify({"status": "error", "message": "Database name is required."}), 400
        
    # Clean name
    name = "".join(c for c in raw_name if c.isalnum() or c in ("_", "-")).strip()
    if not name:
        return jsonify({"status": "error", "message": "Invalid database name characters."}), 400
        
    try:
        select_database(name)
        return jsonify({
            "status": "success",
            "message": f"Database '{name}' created and selected successfully.",
            "active": name
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/databases/delete", methods=["POST"])
def delete_db_route():
    data = request.json or {}
    name = data.get("name", "").strip()
    if not name or name == "default":
        return jsonify({"status": "error", "message": "Cannot delete default or empty database name."}), 400
        
    try:
        db_file = os.path.join(DATABASES_DIR, f"{name}.db")
        if os.path.exists(db_file):
            # If active, switch to default first
            if app_state.get("active_db") == name:
                select_database("default")
            os.remove(db_file)
            
            # Delete models associated with this database
            os.makedirs(MODELS_DIR, exist_ok=True)
            for folder in os.listdir(MODELS_DIR):
                folder_path = os.path.join(MODELS_DIR, folder)
                if os.path.isdir(folder_path):
                    meta_path = os.path.join(folder_path, "meta.json")
                    if os.path.exists(meta_path):
                        try:
                            with open(meta_path, "r", encoding="utf-8") as f:
                                meta = json.load(f)
                            if meta.get("db_name", "default") == name:
                                import shutil
                                shutil.rmtree(folder_path)
                        except Exception as e:
                            print(f"Error deleting associated model checkpoint {folder}: {e}")
            
            return jsonify({"status": "success", "message": f"Database '{name}' and its associated models deleted successfully."})
        else:
            return jsonify({"status": "error", "message": "Database file not found."}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/models/active", methods=["POST"])
def activate_model():
    import json
    data = request.json or {}
    model_id = data.get("model_id")
    if not model_id:
        return jsonify({"status": "error", "message": "Model ID is required."}), 400
        
    dir_path = os.path.join(MODELS_DIR, model_id)
    meta_path = os.path.join(dir_path, "meta.json")
    if not os.path.exists(meta_path):
        return jsonify({"status": "error", "message": "Model not found."}), 404
        
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            
        linked_db = meta.get("db_name", "default")
        if app_state.get("active_db") != linked_db:
            print(f"Auto-switching database to linked DB: {linked_db}")
            select_database(linked_db)
            
        disease_mask = torch.tensor(meta["disease_mask"])
        
        lstm_model = BaselineLSTM(
            len(meta["all_meds"]) + 1,
            len(meta["disease_names"]),
            len(meta["all_meds"]),
            disease_mask,
        )
        lstm_model.load_state_dict(torch.load(os.path.join(dir_path, "lstm.pt"), map_location=torch.device("cpu")))
        
        from module.retail_pipeline import TabularDDPM
        y_dim = len(meta["products"])
        n_steps = int(meta.get("diff_steps", 30))
        
        diffusion_model = TabularDDPM(y_dim=y_dim, num_diseases=len(meta["disease_names"]), n_steps=n_steps)
        diffusion_model.load_state_dict(torch.load(os.path.join(dir_path, "diffusion.pt"), map_location=torch.device("cpu")))
        
        retail_model = ProductRecommenderNN(1 + len(meta["med_to_idx"]), len(meta["prod_to_idx"]))
        retail_model.load_state_dict(torch.load(os.path.join(dir_path, "retail.pt"), map_location=torch.device("cpu")))
        
        encoder = LabelEncoder()
        encoder.classes_ = np.array(meta["encoder_classes"])
        
        app_state["lstm_model"] = lstm_model
        app_state["retail_model"] = retail_model
        app_state["retail_diffusion_model"] = diffusion_model
        app_state["retail_disease_encoder"] = encoder
        app_state["retail_med_to_idx"] = meta["med_to_idx"]
        app_state["retail_prod_to_idx"] = meta["prod_to_idx"]
        app_state["retail_products"] = meta["products"]
        app_state["active_model_id"] = model_id
        
        if app_state["pipeline"] is not None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            app_state["results_df"] = build_results(app_state["pipeline"], lstm_model, device)
            app_state["test_uids"] = app_state["pipeline"].test_seq_df["client_id"].unique().tolist()
            app_state["model_analysis_plot"] = plot_medication_analysis(
                app_state["pipeline"],
                app_state["results_df"],
                app_state["test_uids"][0] if app_state["test_uids"] else None,
            )
            
        return jsonify({
            "status": "success", 
            "message": f"Model {model_id} activated successfully. Linked DB: {linked_db}.",
            "metrics": meta["metrics"],
            "active_db": linked_db
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"Load error: {str(e)}"}), 500


@app.route("/api/models/<model_id>", methods=["DELETE"])
def delete_model(model_id):
    dir_path = os.path.join(MODELS_DIR, model_id)
    if not os.path.exists(dir_path):
        return jsonify({"status": "error", "message": "Model not found."}), 404
    try:
        import shutil
        shutil.rmtree(dir_path)
        if app_state.get("active_model_id") == model_id:
            app_state["lstm_model"] = None
            app_state["retail_model"] = None
            app_state["retail_diffusion_model"] = None
            app_state["retail_disease_encoder"] = None
            app_state["active_model_id"] = ""
        return jsonify({"status": "success", "message": "Model deleted successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=5001)
