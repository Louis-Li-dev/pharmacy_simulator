


import matplotlib.pyplot as plt
import numpy as np
import torch
def plot_client_comparison(client_id, configs, test_df, all_meds):
    """
    Visualizes medication and timing predictions dynamically based on provided configs.
    configs format: [(results_tuple, "Plot Title"), ...]
    """
    num_plots = len(configs)
    fig, axes = plt.subplots(num_plots, 1, figsize=(16, 6 * num_plots), sharex=False)
    
    # Ensure axes is iterable even if there's only 1 plot
    if num_plots == 1:
        axes = [axes]
        
    client_data = test_df[test_df['client_id'] == client_id]
    if client_data.empty:
        return print(f"Client {client_id} not found in dataset.")
    
    disease = client_data['disease'].iloc[0]
        
    for ax, (results, title) in zip(axes, configs):
        pm, tm, pg, tg, cids, tdays = results
        
        mask = (cids == client_id)
        if not mask.any():
            ax.set_title(f"{title}: No data found for ID {client_id}")
            continue
            
        c_pm, c_tm, c_pg, c_tg, c_tdays = pm[mask], tm[mask], pg[mask], tg[mask], tdays[mask]
        
        sort_idx = np.argsort(c_tdays)
        c_pm, c_tm, c_pg, c_tg, c_tdays = c_pm[sort_idx], c_tm[sort_idx], c_pg[sort_idx], c_tg[sort_idx], c_tdays[sort_idx]
        
        plot_t_days, plot_p_days, text_t_meds, text_p_meds = [], [], [], []
        
        for i in range(len(c_tdays)):
            t_day = c_tdays[i]
            true_gap = c_tg[i].item()
            pred_gap = c_pg[i].item()
            
            plot_t_days.append(t_day)
            plot_p_days.append((t_day - true_gap) + pred_gap)
            
            # Predict medications (Sigmoid > 0.5 threshold logic)
            p_probs = torch.sigmoid(c_pm[i])
            p_idx = (p_probs > 0.5).nonzero(as_tuple=False).squeeze(-1)
            text_p_meds.append("\n".join([all_meds[idx] for idx in p_idx]) or "None")
            
            t_idx = c_tm[i].nonzero(as_tuple=False).squeeze(-1)
            text_t_meds.append("\n".join([all_meds[idx] for idx in t_idx]) or "None")

        ax.set_title(f"{title} | Client: {client_id} (Disease: {disease})", fontweight='bold', fontsize=14)
        ax.axhline(1, color='black', linewidth=0.8, alpha=0.3, zorder=1)
        ax.axhline(0, color='black', linewidth=0.8, alpha=0.3, zorder=1)
        
        for i in range(len(plot_t_days)):
            color = 'blue' if abs(plot_t_days[i] - plot_p_days[i]) < 5 else 'orange'
            ax.plot([plot_t_days[i], plot_p_days[i]], [1, 0], color=color, linestyle='--', alpha=0.6, zorder=2)
            
            ax.scatter(plot_t_days[i], 1, color='forestgreen', s=120, edgecolors='white', zorder=3, label='Actual' if i==0 else "")
            ax.text(plot_t_days[i], 1.15, text_t_meds[i], ha='center', va='bottom', fontsize=9, 
                    bbox=dict(facecolor='white', edgecolor='forestgreen', alpha=0.9, boxstyle='round,pad=0.3'))
            
            ax.scatter(plot_p_days[i], 0, color='crimson', s=120, edgecolors='white', zorder=3, label='Predicted' if i==0 else "")
            ax.text(plot_p_days[i], -0.15, text_p_meds[i], ha='center', va='top', fontsize=9, 
                    bbox=dict(facecolor='white', edgecolor='crimson', alpha=0.9, boxstyle='round,pad=0.3'))

        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Model Prediction', 'Ground Truth'], fontweight='bold')
        ax.set_ylim(-1.2, 2.2) 
        ax.set_xlabel("Timeline (Days)", fontsize=10)
        ax.grid(axis='x', linestyle=':', alpha=0.6)
        
        all_x = plot_t_days + plot_p_days
        ax.set_xlim(min(all_x) - 30, max(all_x) + 30)

    plt.tight_layout()
    plt.show()