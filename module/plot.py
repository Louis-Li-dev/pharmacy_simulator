import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_client_comparison(client_id, configs, test_df, all_meds):
    """
    動態視覺化個別個案（Client）在各模型下的預測領藥時間與藥物組合比對圖。
    
    參數:
        client_id (str): 個案編號 (例如: "C_001")
        configs (list): 包含 [(results_tuple, "模型標題"), ...] 的設定清單
        test_df (pd.DataFrame): 測試集資料表
        all_meds (list): 所有處方藥物名稱清單
    """
    num_plots = len(configs)
    fig, axes = plt.subplots(num_plots, 1, figsize=(16, 6 * num_plots), sharex=False)
    
    # 若僅有單一模型圖表，統一轉為可疊代的列表
    if num_plots == 1:
        axes = [axes]
        
    client_data = test_df[test_df['client_id'] == client_id]
    if client_data.empty:
        return print(f"未在資料集中找到個案 {client_id}。")
    
    disease = client_data['disease'].iloc[0]
        
    for ax, (results, title) in zip(axes, configs):
        pm, tm, pg, tg, cids, tdays = results
        
        mask = (cids == client_id)
        if not mask.any():
            ax.set_title(f"{title}: 個案 {client_id} 無可用數據")
            continue
            
        c_pm, c_tm, c_pg, c_tg, c_tdays = pm[mask], tm[mask], pg[mask], tg[mask], tdays[mask]
        
        # 依領藥時間天數排序
        sort_idx = np.argsort(c_tdays)
        c_pm, c_tm, c_pg, c_tg, c_tdays = c_pm[sort_idx], c_tm[sort_idx], c_pg[sort_idx], c_tg[sort_idx], c_tdays[sort_idx]
        
        plot_t_days, plot_p_days, text_t_meds, text_p_meds = [], [], [], []
        
        for i in range(len(c_tdays)):
            t_day = c_tdays[i]
            true_gap = c_tg[i].item()
            pred_gap = c_pg[i].item()
            
            plot_t_days.append(t_day)
            plot_p_days.append((t_day - true_gap) + pred_gap)
            
            # 依 Sigmoided 機率 > 0.5 門檻推論預測藥物
            p_probs = torch.sigmoid(c_pm[i])
            p_idx = (p_probs > 0.5).nonzero(as_tuple=False).squeeze(-1)
            text_p_meds.append("\n".join([all_meds[idx] for idx in p_idx]) or "無")
            
            t_idx = c_tm[i].nonzero(as_tuple=False).squeeze(-1)
            text_t_meds.append("\n".join([all_meds[idx] for idx in t_idx]) or "無")

        # 繪製個案預測時間軸與地面真值 (Ground Truth) 比對
        ax.set_title(f"{title} | 個案: {client_id} (疾病: {disease})", fontweight='bold', fontsize=14)
        ax.axhline(1, color='black', linewidth=0.8, alpha=0.3, zorder=1)
        ax.axhline(0, color='black', linewidth=0.8, alpha=0.3, zorder=1)
        
        for i in range(len(plot_t_days)):
            # 若預測天數誤差在 5 天內使用藍色虛線，否則使用橘色虛線標示
            color = 'blue' if abs(plot_t_days[i] - plot_p_days[i]) < 5 else 'orange'
            ax.plot([plot_t_days[i], plot_p_days[i]], [1, 0], color=color, linestyle='--', alpha=0.6, zorder=2)
            
            # 實際領藥時間點 (綠色點)
            ax.scatter(plot_t_days[i], 1, color='forestgreen', s=120, edgecolors='white', zorder=3, label='實際情況' if i==0 else "")
            ax.text(plot_t_days[i], 1.15, text_t_meds[i], ha='center', va='bottom', fontsize=9, 
                    bbox=dict(facecolor='white', edgecolor='forestgreen', alpha=0.9, boxstyle='round,pad=0.3'))
            
            # 模型預測時間點 (紅色點)
            ax.scatter(plot_p_days[i], 0, color='crimson', s=120, edgecolors='white', zorder=3, label='模型預測' if i==0 else "")
            ax.text(plot_p_days[i], -0.15, text_p_meds[i], ha='center', va='top', fontsize=9, 
                    bbox=dict(facecolor='white', edgecolor='crimson', alpha=0.9, boxstyle='round,pad=0.3'))

        ax.set_yticks([0, 1])
        ax.set_yticklabels(['模型預測 (Model Prediction)', '實際領藥 (Ground Truth)'], fontweight='bold')
        ax.set_ylim(-1.2, 2.2) 
        ax.set_xlabel("時間軸 (天數 / Days)", fontsize=10)
        ax.grid(axis='x', linestyle=':', alpha=0.6)
        
        all_x = plot_t_days + plot_p_days
        ax.set_xlim(min(all_x) - 30, max(all_x) + 30)

    plt.tight_layout()
    plt.show()