import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch

class ClinicalSimulationPipeline:
    def __init__(self, n_clients=200, simulation_days=2880, split_day=2340, seed=42):
        """
        Initializes the simulation environment and base parameters.
        """
        self.n_clients = n_clients
        self.simulation_days = simulation_days
        self.split_day = split_day
        self.seed = seed
        
        # Set seeds for reproducibility
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.chronic_diseases = {
            "Hypertension": [
                {"name": "Lisinopril", "M_days": 30},
                {"name": "Amlodipine", "M_days": 30},
                {"name": "HCTZ", "M_days": 60}
            ],
            "Type_2_Diabetes": [
                {"name": "Metformin", "M_days": 60},
                {"name": "Glipizide", "M_days": 30},
                {"name": "Empagliflozin", "M_days": 30},
                {"name": "Insulin", "M_days": 30}
            ],
            "Asthma": [
                {"name": "Inhaled Corticosteroid", "M_days": 60},
                {"name": "Albuterol Rescue", "M_days": 90},
                {"name": "Leukotriene Modifier", "M_days": 30}
            ],
            # --- NEW DISEASES ---
            "Hyperlipidemia": [
                {"name": "Atorvastatin", "M_days": 90}, # Statins usually given in 90-day supplies
                {"name": "Ezetimibe", "M_days": 30},
                {"name": "PCSK9_Inhibitor", "M_days": 30} # Injectables usually 30-day
            ],
            "Rheumatoid_Arthritis": [
                {"name": "Methotrexate", "M_days": 30},
                {"name": "Folic_Acid", "M_days": 90},
                {"name": "Adalimumab", "M_days": 28} # Biologics often follow 4-week (28 day) cycles
            ]
        }
        
        self.disease_names = list(self.chronic_diseases.keys())
        self.all_meds = []
        for meds in self.chronic_diseases.values():
            self.all_meds.extend([m["name"] for m in meds])
            
        # Data storage attributes
        self.df = None
        self.sequence_df = None
        self.train_seq_df = None
        self.test_seq_df = None
        self.disease_mask = None

    def generate_pathways(self):
        """
        Generates dynamic clinical pathways for clients based on disease progression.
        """
        client_profiles = []
        for client_id in range(1, self.n_clients + 1):
            client_profiles.append({
                "client_id": f"C_{client_id:03d}",
                "age": int(np.clip(np.random.normal(58, 16), 18, 95)),
                "lam_doctor": np.random.uniform(0.5, 4.0),
                "lam_pharmacy": np.random.uniform(1.5, 4.5),
                "disease": np.random.choice(self.disease_names),
                "progression_rate": np.random.uniform(0.02, 0.15) 
            })

        records = []
        for client in client_profiles:
            current_day = np.random.randint(0, 30) 
            disease_med_pool = self.chronic_diseases[client["disease"]]
            current_stage = 0 
            
            # --- UPDATED PATHWAYS ---
            if client["disease"] == "Hypertension":
                pathway = [["Lisinopril"], ["Lisinopril", "Amlodipine"], ["Lisinopril", "Amlodipine", "HCTZ"]]
            elif client["disease"] == "Type_2_Diabetes":
                pathway = [["Metformin"], ["Metformin", "Glipizide"], ["Metformin", "Glipizide", "Empagliflozin"], ["Metformin", "Empagliflozin", "Insulin"]]
            elif client["disease"] == "Hyperlipidemia":
                pathway = [["Atorvastatin"], ["Atorvastatin", "Ezetimibe"], ["Atorvastatin", "Ezetimibe", "PCSK9_Inhibitor"]]
            elif client["disease"] == "Rheumatoid_Arthritis":
                pathway = [["Methotrexate", "Folic_Acid"], ["Methotrexate", "Folic_Acid", "Adalimumab"]]
            else:
                pathway = None # Handled dynamically below for Asthma
            
            while current_day <= self.simulation_days:
                pharmacy_delay = min(3, np.random.poisson(client['lam_pharmacy'])) 
                
                # --- NEW: Random Delay Noise ---
                # 3% chance of a significant delay (insurance auth, out of town, forgot)
                pharmacy_visit_day = current_day + pharmacy_delay
                if pharmacy_visit_day > self.simulation_days: 
                    break
                    
                if client["disease"] == "Asthma":
                    is_winter = np.sin((current_day / 365.0) * 2 * np.pi) > 0.5
                    if is_winter:
                        prescribed_meds_names = ["Inhaled Corticosteroid", "Albuterol Rescue", "Leukotriene Modifier"]
                    else:
                        prescribed_meds_names = ["Inhaled Corticosteroid"]
                else:
                    if np.random.rand() < client["progression_rate"] and current_stage < len(pathway) - 1:
                        current_stage += 1
                    prescribed_meds_names = pathway[current_stage]

                prescribed_meds = [m for m in disease_med_pool if m["name"] in prescribed_meds_names]
                m_days = max(m["M_days"] for m in prescribed_meds)
                
                records.append({
                    "client_id": client["client_id"],
                    "age": client["age"],
                    "disease": client["disease"],
                    "doctor_visit_day": current_day,
                    "pharmacy_report_day": pharmacy_visit_day,
                    "m_days": m_days,
                    "meds_received": prescribed_meds_names.copy()
                })
                current_day = pharmacy_visit_day + m_days + np.random.poisson(client['lam_doctor'])

        self.df = pd.DataFrame(records)
        print("1. Dynamic Clinical Pathway Generation Complete.")

    def visualize(self, sample_size=15):
        """
        Plots the Omniscient and Latent representations of patient pathways.
        """
        if self.df is None:
            raise ValueError("Dataset not generated. Run generate_pathways() first.")
            
        print(">> Displaying Visualizations (Close the first window to see the second...)")

        sample_clients = self.df['client_id'].unique()[:sample_size]
        raw_sample = self.df[self.df['client_id'].isin(sample_clients)]

        # --- PLOT A: Omniscient View ---
        fig, ax = plt.subplots(figsize=(14, 8))
        y_labels = []

        for i, client in enumerate(sample_clients):
            patient_data = raw_sample[raw_sample['client_id'] == client]
            disease = patient_data['disease'].iloc[0].replace('_', ' ')
            y_labels.append(f"{client}\n({disease})")
            
            for _, row in patient_data.iterrows():
                doc_day = row['doctor_visit_day']
                rx_day = row['pharmacy_report_day']
                end_day = rx_day + row['m_days']
                
                ax.plot([rx_day, end_day], [i, i], color='darkgrey', linewidth=6, solid_capstyle='butt')
                ax.plot(doc_day, i, marker='o', color='black', markersize=8) 
                ax.plot(rx_day, i, marker='s', color='black', markersize=8)  

        ax.set_yticks(range(len(sample_clients)))
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("Day of Simulation")
        ax.set_title("Version 1: Omniscient Timeline (Includes Doctor Visits)")
        ax.xaxis.grid(True, linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Doctor Visit', markerfacecolor='black', markersize=10),
            Line2D([0], [0], marker='s', color='w', label='Pharmacy Visit', markerfacecolor='black', markersize=10),
            Line2D([0], [0], color='darkgrey', lw=6, label='Medication Coverage')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True, edgecolor='black')
        plt.tight_layout()
        # plt.show()

        # --- PLOT B: Model's View ---
        fig, ax = plt.subplots(figsize=(14, 8))

        for i, client in enumerate(sample_clients):
            patient_data = raw_sample[raw_sample['client_id'] == client]
            for _, row in patient_data.iterrows():
                rx_day = row['pharmacy_report_day']
                end_day = rx_day + row['m_days']
                
                ax.plot([rx_day, end_day], [i, i], color='darkgrey', linewidth=6, solid_capstyle='butt')
                ax.plot(rx_day, i, marker='s', color='black', markersize=8)  

        ax.set_yticks(range(len(sample_clients)))
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("Day of Simulation")
        ax.set_title("Version 2: The Model's View (Doctor Visits are latent/hidden)")
        ax.xaxis.grid(True, linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)

        legend_elements_2 = [
            Line2D([0], [0], marker='s', color='w', label='Pharmacy Visit', markerfacecolor='black', markersize=10),
            Line2D([0], [0], color='darkgrey', lw=6, label='Medication Coverage')
        ]
        ax.legend(handles=legend_elements_2, loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True, edgecolor='black')
        plt.tight_layout()
        # plt.show()

    def preprocess_sequences(self, seq_len=5):
        """
        Calculates visitation gaps and structures data into train/test sequence datasets.
        """
        if self.df is None:
            raise ValueError("Dataset not generated. Run generate_pathways() first.")
            
        print("2. Formatting Sequence History [visit_{t-k, t} -> visit_{t+1}]...")

        # Calculate gaps
        self.df = self.df.sort_values(by=['client_id', 'pharmacy_report_day'])
        self.df['prev_report_day'] = self.df.groupby('client_id')['pharmacy_report_day'].shift(1)
        self.df['prev_m_days'] = self.df.groupby('client_id')['m_days'].shift(1)

        self.df['gap'] = self.df['pharmacy_report_day'] - (self.df['prev_report_day'] + self.df['prev_m_days'])
        self.df['gap'] = self.df['gap'].clip(lower=0).fillna(0) 

        # Build sequence datasets
        sequences = []
        for client_id, group in self.df.groupby('client_id'):
            group = group.sort_values('pharmacy_report_day')
            visits = group.to_dict('records')
            
            for i in range(1, len(visits)):
                target_visit = visits[i]
                start_idx = max(0, i - seq_len)
                history_visits = visits[start_idx:i]
                
                history_formatted = [{
                    "M_gap": v['gap'],
                    "meds": v['meds_received'],
                    "m_days": v['m_days']
                } for v in history_visits]
                
                target_formatted = {
                    "M_gap": target_visit['gap'],
                    "meds": target_visit['meds_received']
                }
                
                sequences.append({
                    "client_id": client_id,
                    "age": int(target_visit.get("age", 45)),
                    "disease": target_visit['disease'],
                    "target_pharmacy_day": target_visit['pharmacy_report_day'],
                    "history_window": history_formatted,
                    "target": target_formatted
                })
                
        self.sequence_df = pd.DataFrame(sequences)

        # Time split
        self.train_seq_df = self.sequence_df[self.sequence_df['target_pharmacy_day'] < self.split_day].copy()
        self.test_seq_df = self.sequence_df[self.sequence_df['target_pharmacy_day'] >= self.split_day].copy()
        
        # Example Output Log
        print("\n--- Example Sequence Output (Row 0) ---")
        first_row = self.sequence_df.iloc[0]
        print(f"Client: {first_row['client_id']} ({first_row['disease']})")
        print("History Window (t-k to t):")
        for idx, visit in enumerate(first_row['history_window']):
            print(f"  Step {idx+1}: Gap = {visit['M_gap']} days | Meds = {visit['meds']}")
        print(f"Target (t+1):\n  Gap = {first_row['target']['M_gap']} days | Meds = {first_row['target']['meds']}")

    def create_mappings(self):
        """
        Creates index mappings for diseases/medications and computes the disease tensor mask.
        """
        self.disease_to_idx = {d: i for i, d in enumerate(self.disease_names)}
        self.med_to_idx = {m: i for i, m in enumerate(self.all_meds)}

        num_diseases = len(self.disease_names)
        num_meds = len(self.all_meds)

        self.disease_mask = torch.zeros((num_diseases, num_meds))
        for d_name, meds in self.chronic_diseases.items():
            d_idx = self.disease_to_idx[d_name]
            for med in meds:
                self.disease_mask[d_idx, self.med_to_idx[med["name"]]] = 1.0

        print(f"\n>> Disease Mask Created: {self.disease_mask.shape} (Diseases x Meds)")
        
        if self.train_seq_df is not None and self.test_seq_df is not None:
            print(f"3. Data split complete. Train Rows: {len(self.train_seq_df)} | Test Rows: {len(self.test_seq_df)}")


# ==========================================
# 3. CALLING SCRIPT
# ==========================================
if __name__ == "__main__":
    # 1. Instantiate the Pipeline class
    pipeline = ClinicalSimulationPipeline(
        n_clients=200, 
        simulation_days=12 * 30 * 8, 
        split_day=12 * 30 * 6 + 12 * 2
    )

    # 2. Run the Generation Phase
    pipeline.generate_pathways()

    # 3. View the Visualizations 
    # (Will halt execution until the plot windows are closed)
    pipeline.visualize(sample_size=15)

    # 4. Preprocess Data and build time-series sequences
    pipeline.preprocess_sequences(seq_len=5)

    # 5. Build ML Tensor Mappings / Masks
    pipeline.create_mappings()
