import base64
import io
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, Response, jsonify, render_template, request
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from module.data import ClinicalDataset
from module.model import BaselineLSTM
from module.retail_pipeline import BasketDataset, ProductRecommenderNN, train_diffusion


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

app_state = {
    "pipeline": None,
    "train_loader": None,
    "test_loader": None,
    "lstm_model": None,
    "retail_model": None,
    "retail_diffusion_model": None,
    "retail_disease_encoder": None,
    "retail_med_to_idx": None,
    "retail_prod_to_idx": None,
    "retail_products": [],
    "results_df": None,
    "test_uids": [],
    "training_log": [],
    "model_analysis_plot": None,
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

        sequences = []
        for client_id, group in self.df.groupby("client_id"):
            visits = group.sort_values("pharmacy_report_day").to_dict("records")
            for idx in range(1, len(visits)):
                target_visit = visits[idx]
                history_visits = visits[max(0, idx - seq_len):idx]
                sequences.append(
                    {
                        "client_id": client_id,
                        "age": int(target_visit["age"]),
                        "disease": target_visit["disease"],
                        "target_pharmacy_day": int(target_visit["pharmacy_report_day"]),
                        "history_window": [
                            {
                                "M_gap": float(visit["gap"]),
                                "meds": visit["meds_received"],
                                "m_days": int(visit["m_days"]),
                            }
                            for visit in history_visits
                        ],
                        "target": {
                            "M_gap": float(target_visit["gap"]),
                            "meds": target_visit["meds_received"],
                            "products": target_visit["retail_products"],
                        },
                    }
                )

        self.sequence_df = pd.DataFrame(sequences)
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
    model = train_diffusion(
        diffusion_df,
        pipeline.products,
        len(encoder.classes_),
        {"batch_size": 128, "lr": 1e-3, "diff_epochs": epochs, "diff_steps": steps},
    )
    return model, encoder


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
        rows.append(
            {
                "client_id": row["client_id"],
                "age": int(row["age"]),
                "disease": row["disease"],
                "target_pharmacy_day": int(round(float(target_days[idx]))),
                "predicted_pharmacy_day": int(round(float(target_days[idx] - true_gaps[idx].item() + pred_gaps[idx].item()))),
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
                "predicted_gap": int(round(predicted_gap)),
                "predicted_meds": "|".join(predicted_meds),
                "recommended_products": "|".join(item["product"] for item in recommendations),
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
    try:
        df = pd.read_csv(file)
        missing = sorted(REQUIRED_COLUMNS - set(df.columns))
        if missing:
            return jsonify({"status": "error", "message": f"Missing columns: {', '.join(missing)}"}), 400
        seq_len = int(request.form.get("seq_len", 5))
        split_ratio = float(request.form.get("split_ratio", 0.8))
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
    lstm_model.fit(app_state["train_loader"], epochs=epochs, lr=lr, device=device)
    logs.append(f"LSTM complete: {epochs} epochs.")

    retail_model, retail_logs = train_retail_nn(pipeline, epochs=retail_epochs)
    logs.extend(retail_logs)
    logs.append("Training retail diffusion model.")
    diffusion_model, encoder = train_retail_diffusion(pipeline, epochs=diff_epochs, steps=diff_steps)
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

    return jsonify(
        {
            "status": "success",
            "logs": logs,
            "metrics": metrics_from_results(app_state["results_df"]),
            "test_uids": app_state["test_uids"],
            "model_analysis_plot": app_state["model_analysis_plot"],
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
    csv_data = app_state["results_df"].to_csv(index=False)
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=prediction_results.csv"},
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=5001)
