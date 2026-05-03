import os
import io
import base64
import torch
import matplotlib
matplotlib.use('Agg')  # 必須在導入 pyplot 之前調用
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
import os
import sys

parent_dir = os.path.join(os.getcwd(), '..')
sys.path.append(parent_dir)
# 確保從您的腳本中匯入對比繪圖函數
from module.data_generator import ClinicalSimulationPipeline
from module.model import  BaselineLSTM
from module.data import ClinicalDataset
from module.plot import plot_client_comparison
from torch.utils.data import DataLoader

app = Flask(__name__)

# 全域狀態暫存
app_state = {
    "pipeline": None,
    "train_loader": None,
    "test_loader": None,
    "model": None,         # 新增：儲存訓練好的模型
    "test_uids": []        # 新增：儲存測試集中的可用病患 ID 名單
}

def get_plot_as_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.clf()
    plt.close('all')
    return img_base64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_data', methods=['POST'])
def generate_data():
    try:
        data = request.json
        n_clients = int(data.get('n_clients', 100))
        sim_days = int(data.get('simulation_days', 2880))
        seq_len = int(data.get('seq_len', 5))
        split_day = int(sim_days * 0.8) # 自動計算 80% 作為切分點
        
        pipeline = ClinicalSimulationPipeline(n_clients=n_clients, simulation_days=sim_days, split_day=split_day)
        pipeline.generate_pathways()
        
        plt.figure(figsize=(14, 8))
        pipeline.visualize(sample_size=10) 
        data_plot_base64 = get_plot_as_base64()

        pipeline.preprocess_sequences(seq_len=seq_len)
        pipeline.create_mappings()
        
        train_dataset = ClinicalDataset(pipeline.train_seq_df, pipeline.all_meds, pipeline.disease_to_idx, seq_len=seq_len)
        test_dataset = ClinicalDataset(pipeline.test_seq_df, pipeline.all_meds, pipeline.disease_to_idx, seq_len=seq_len)
        
        # 原本的程式碼
        app_state["train_loader"] = DataLoader(train_dataset, batch_size=32, shuffle=True)
        app_state["test_loader"] = DataLoader(test_dataset, batch_size=32, shuffle=False)
        app_state["pipeline"] = pipeline
        
        # 【新增這兩行】：記住使用者的生成設定
        app_state["sim_days"] = sim_days
        app_state["seq_len"] = seq_len

        return jsonify({"status": "success", "data_plot": data_plot_base64})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        if app_state["pipeline"] is None:
            return jsonify({"status": "error", "message": "請先生成資料！"})

        data = request.json
        epochs = int(data.get('epochs', 5))
        lr = float(data.get('lr', 0.005))
        
        pipeline = app_state["pipeline"]
        train_loader = app_state["train_loader"]
        test_loader = app_state["test_loader"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = len(pipeline.all_meds) + 1
        lstm_model = BaselineLSTM(input_dim, len(pipeline.disease_names), len(pipeline.all_meds), pipeline.disease_mask)
        lstm_model.fit(train_loader, epochs=epochs, lr=lr, device=device)
        
        # 儲存模型與可用 UID
        app_state["model"] = lstm_model
        app_state["test_uids"] = pipeline.test_seq_df['client_id'].unique().tolist()
        
        res_lstm_masked = lstm_model.predict(test_loader, device=device, apply_mask=True)
        plot_configs = [(res_lstm_masked, "Baseline LSTM (Masked)")]
        
        sample_client = app_state["test_uids"][0] if app_state["test_uids"] else None
        if sample_client:
            plot_client_comparison(sample_client, plot_configs, pipeline.test_seq_df, pipeline.all_meds)
        
        result_plot_base64 = get_plot_as_base64()
        err = torch.abs(res_lstm_masked[2] - res_lstm_masked[3]).mean().item()
        
        return jsonify({
            "status": "success",
            "result_plot": result_plot_base64,
            "metrics": {"平均天數誤差 (Day Error)": f"±{err:.2f} 天"},
            "test_uids": app_state["test_uids"] # 回傳可用名單給前端儀表板
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# ==========================================
# 新增 API: 專門提供單一病患的 JSON 數據供 ECharts 渲染
# ==========================================
@app.route('/api/get_patient/<uid>')
def get_patient(uid):
    if app_state["pipeline"] is None or app_state["model"] is None:
        return jsonify({"error": "模型尚未訓練完成"}), 400

    pipeline = app_state["pipeline"]
    model = app_state["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 真實完整時間線 (上帝視角)
    p_df = pipeline.df[pipeline.df['client_id'] == uid].sort_values('pharmacy_report_day')
    if p_df.empty: return jsonify({"error": "找不到病患資料"}), 404

    raw_timeline = [
        {"doc_day": int(row['doctor_visit_day']), "rx_day": int(row['pharmacy_report_day']), "m_days": int(row['m_days'])} 
        for _, row in p_df.iterrows()
    ]

    # 2. 找出測試集預測目標
    test_seq = pipeline.test_seq_df[pipeline.test_seq_df['client_id'] == uid]
    if test_seq.empty: return jsonify({"error": "病患不在測試集中"}), 404
    target_row = test_seq.iloc[-1] 
    target_rx_day = int(target_row['target_pharmacy_day'])
    
    # 2. 找到這一次真實的看診日
    target_visit_raw = p_df[p_df['pharmacy_report_day'] == target_rx_day].iloc[0]
    target_doc_day = int(target_visit_raw['doctor_visit_day'])
    
    # 3. 【關鍵修復】：不要用減法算！直接從資料表抓出「上一次」的真實紀錄
    prev_visit_raw = p_df[p_df['pharmacy_report_day'] < target_rx_day].iloc[-1]
    base_day = int(prev_visit_raw['pharmacy_report_day'])
    last_m_days = int(prev_visit_raw['m_days'])
    
    # 4. 正確計算兩個階段的 Gap (保證絕對是正數)
    med_end_day = base_day + last_m_days
    doc_gap = target_doc_day - med_end_day 
    rx_delay = target_rx_day - target_doc_day

    # 構建模型視窗輸入
    model_input = [{"day": rx['rx_day'], "m_days": rx['m_days']} for rx in raw_timeline if rx['rx_day'] <= base_day]
    last_m_days = int(model_input[-1]['m_days']) if model_input else 0
    med_end_day = base_day + last_m_days

    # 【新增】：精確計算兩個階段的 Gap
    doc_gap = target_doc_day - med_end_day # 從藥吃完到去看診的空窗期
    rx_delay = target_rx_day - target_doc_day # 從看診到實際領藥的延遲

    # 3. 單獨推論此筆數據
    seq_len = len(target_row['history_window'])
    mini_dataset = ClinicalDataset(test_seq.tail(1), pipeline.all_meds, pipeline.disease_to_idx, seq_len=seq_len)
    mini_loader = DataLoader(mini_dataset, batch_size=1)
    res = model.predict(mini_loader, device=device, apply_mask=True)
    pred_gap = res[2][0].item()

    # 4. 歷史分佈
    gaps = [float(g) for g in p_df['gap'].tolist() if g > 0]
    rx_delays_history = [int(rx['rx_day'] - rx['doc_day']) for rx in raw_timeline]

    return jsonify({
        "uid": uid,
        "disease": target_row['disease'],
        "raw_timeline": raw_timeline,
        "model_input": model_input,
        "prediction": {"base_day": base_day, "pred_gap": round(pred_gap, 1), "last_m_days": last_m_days},
        "target": {
            "real_gap": round(float(target_row['target']['M_gap']), 1),
            "doc_gap": int(doc_gap),      # 新增
            "rx_delay": int(rx_delay)     # 新增
        },
        "distributions": {"gaps": gaps},
        "rx_delays": rx_delays_history
    })

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000)