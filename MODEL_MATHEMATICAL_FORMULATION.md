# 📐 Pharmacy Simulator - 系統模型數學定義與理論公式說明書 (Mathematical Formulations & Model Definitions)

本說明文件依據系統實作（`module/model.py`與`module/retail_pipeline.py`）之數學理論與演算法邏輯編寫，提供慢性病連續處方用藥預測模型（CVAE, Baseline LSTM, Statistical Model）及零售連帶商品推薦模型（Tabular DDPM, ProductRecommenderNN）之嚴謹數學定義、輸入輸出張量表示、前向與反向傳播公式、損失函數與評估指標。

---

## 1. 符號與資料表示法 (Mathematical Notations & Representations)

在系統中，定義以下符號與變數特徵表示法：

### 1.1 臨床處方領藥序列 (Clinical Refill Sequence)
定義單一個案的歷史領藥序列為 $X = [\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T] \in \mathbb{R}^{T \times (M + 1)}$，其中 $T$ 為序列長度（系統設 $T=10$）：
- **時間步特徵向量**：$\mathbf{x}_t = [\mathbf{m}_t ; d_t] \in \mathbb{R}^{M + 1}$
- **處方藥物向量**：$\mathbf{m}_t \in \{0, 1\}^M$ 表示在第 $t$ 次領藥時，開立之 $M$ 種候選處方藥物的一熱（One-hot）或多標籤二元向量。
- **領藥天數向量**：$d_t = \frac{\text{m\_days}}{100} \in [0, 1]$ 表示當次開立處方箋的藥物供給天數（已除以基底天數 100 進行正規化）。

### 1.2 疾病條件與遮罩 (Disease Condition & Masking)
- **疾病類別**：$c \in \{0, 1, \dots, C-1\}$（系統定義 $C=5$ 大慢性疾病）。
- **疾病嵌入向量 (Disease Embedding)**：$\mathbf{e}_c = \mathbf{E}(c) \in \mathbb{R}^{d_e}$，其中 $\mathbf{E} \in \mathbb{R}^{C \times 16}$ 為可學習之疾病嵌入矩陣。
- **疾病可行用藥遮罩 (Disease Masking Vector)**：$\mathbf{M}_c \in \{0, 1\}^M$。若藥物 $i$ 屬於疾病 $c$ 的合法處方範疇，則 $M_{c, i} = 1$；否則 $M_{c, i} = 0$。

### 1.3 預測目標 (Prediction Targets)
- **預測處方藥物**：$\mathbf{y}_m \in \{0, 1\}^M$（下次領藥時實際開立之藥物多標籤）。
- **預測領藥間隔天數**：$y_g = \frac{\Delta t}{100} \in [0, 1]$（下次實際領藥與本次領藥之間隔天數）。

---

## 2. 慢性病領藥預測模型 (Clinical Medication Forecasting Models)

### 2.1 條件式變異自編碼器模型 (Conditional VAE - CVAE)

為解決慢性病處方用藥的極度稀疏性與隨機性，CVAE 模型引入潛在空間 (Latent Space) 變數 $\mathbf{z} \in \mathbb{R}^d$（系統設 $d=32$）。

#### (A) 條件編碼器 (Conditional Encoder)
編碼器接收序列輸入 $X$ 與疾病條件 $\mathbf{e}_c$：
$$\tilde{\mathbf{x}}_t = [\mathbf{x}_t ; \mathbf{e}_c] \in \mathbb{R}^{(M + 1 + 16)}$$
$$\mathbf{h}_T = \text{LSTM}_E(\tilde{\mathbf{x}}_1, \tilde{\mathbf{x}}_2, \dots, \tilde{\mathbf{x}}_T)$$

由最終狀態 $\mathbf{h}_T$ 計算潛在高斯分布之參數（均值 $\boldsymbol{\mu}$ 與對數方差 $\ln \boldsymbol{\sigma}^2$）：
$$\boldsymbol{\mu} = \mathbf{W}_\mu \mathbf{h}_T + \mathbf{b}_\mu, \quad \ln \boldsymbol{\sigma}^2 = \mathbf{W}_\sigma \mathbf{h}_T + \mathbf{b}_\sigma$$

#### (B) 重參數化技巧 (Reparameterization Trick)
為保證梯度可逆向傳播，從標準常態分布中採樣噪聲 $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$：
$$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$$

#### (C) 條件解碼器與跳躍連接 (Decoder with Skip Connection)
解碼器接收潛在變數 $\mathbf{z}$ 與疾病條件 $\mathbf{e}_c$：
$$\mathbf{h}_{\text{dec}} = \text{MLP}_{\text{dec}}([\mathbf{z} ; \mathbf{e}_c])$$

**時間記憶跳躍連接 (Temporal Skip Connection)**：
$$\mathbf{h}_{\text{combined}} = \mathbf{h}_{\text{dec}} + \mathbf{h}_T$$
融合潛在生成變數與 LSTM 原始時間序列記憶後輸出標頭：
$$\hat{\mathbf{z}}_m = \mathbf{W}_m \mathbf{h}_{\text{combined}} + \mathbf{b}_m, \quad \hat{y}_g = \text{Softplus}(\mathbf{W}_g \mathbf{h}_{\text{combined}} + b_g)$$

#### (D) 損失函數與 Focal Loss
為解決稀疏藥物預測中的正負樣本極度不平衡問題，採用 **Focal Loss**：
$$\mathcal{L}_{\text{Focal}}(\hat{\mathbf{z}}_m, \mathbf{y}_m) = -\frac{1}{M} \sum_{i=1}^M \left[ \alpha (1 - p_{t, i})^\gamma \ln(p_{t, i}) \right]$$
其中 $p_{t, i} = \sigma(\hat{z}_{m, i})$ 若 $y_{m, i}=1$，否則 $p_{t, i} = 1 - \sigma(\hat{z}_{m, i})$。超參數設為 $\alpha = 0.25, \gamma = 2.0$。

**KL 散度 (Kullback-Leibler Divergence)**：
$$D_{\text{KL}}\left(q_\phi(\mathbf{z} \mid X, c) \parallel p(\mathbf{z})\right) = -\frac{1}{2} \sum_{j=1}^d \left( 1 + \ln \sigma_j^2 - \mu_j^2 - \sigma_j^2 \right)$$

**CVAE 總目標函數**：
$$\mathcal{L}_{\text{CVAE}} = 50.0 \cdot \mathcal{L}_{\text{Focal}} + 10.0 \cdot \text{MSE}(\hat{y}_g, y_g) + 0.001 \cdot D_{\text{KL}}$$

#### (E) 多重採樣集成推論 (Ensemble Sampling Inference)
推論時，固定輸入 $X$ 與 $c$，重複採樣 $S=10$ 次潛在變數 $\mathbf{z}^{(s)} \sim q_\phi(\mathbf{z} \mid X, c)$，求取預測機率之集體平均：
$$\bar{p}_i = \frac{1}{S} \sum_{s=1}^S \sigma\left( \hat{z}_{m, i}^{(s)} \right)$$

---

### 2.2 帶有疾病遮罩的序列 LSTM 模型 (Masked Sequence LSTM Model)

#### (A) 網絡架構與前向傳播 (Forward Pass)
LSTM 處理時間序列 $X = [\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T]$ 並擷取隱藏狀態：
$$(\mathbf{h}_1, \mathbf{h}_2, \dots, \mathbf{h}_T) = \text{LSTM}(\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T; \Theta_{\text{LSTM}})$$

將最後一時間步之隱藏狀態 $\mathbf{h}_T \in \mathbb{R}^{h}$ 與疾病嵌入向量 $\mathbf{e}_c \in \mathbb{R}^{16}$ 拼接，通過全連接層融合：
$$\mathbf{z}_{\text{shared}} = \text{ReLU}\left( \mathbf{W}_f [\mathbf{h}_T ; \mathbf{e}_c] + \mathbf{b}_f \right) \in \mathbb{R}^h$$

#### (B) 處方藥物與領藥天數分支標頭 (Task Heads)
1. **未遮罩之藥物 Logits**：
   $$\tilde{\mathbf{z}}_m = \mathbf{W}_m \mathbf{z}_{\text{shared}} + \mathbf{b}_m \in \mathbb{R}^M$$

2. **疾病可行性硬遮罩 (Hard Disease Masking)**：
   $$\hat{z}_{m, i} = \begin{cases} \tilde{z}_{m, i}, & \text{if } M_{c, i} = 1 \\ -10^9, & \text{if } M_{c, i} = 0 \end{cases}$$

3. **領藥間隔天數預測**：
   $$\hat{y}_g = \text{Softplus}\left( \mathbf{W}_g \mathbf{z}_{\text{shared}} + b_g \right)$$

#### (C) 聯合損失函數 (Joint Loss Function)
$$\mathcal{L}_{\text{LSTM}} = \mathcal{L}_{\text{BCEWithLogits}}(\hat{\mathbf{z}}_m, \mathbf{y}_m) + 20.0 \cdot \text{MSE}(\hat{y}_g, y_g)$$

---

### 2.3 統計學預測模型 (Statistical Model: Logistic + Ridge AR)

1. **藥物預測 (Multi-Output Logistic Regression)**：
   將輸入展開為一維特徵向量 $\mathbf{f} = [\text{vec}(X) ; c]$，對 $M$ 種藥物建立獨立的 Logistic 分類器：
   $$P(y_{m, i} = 1 \mid \mathbf{f}) = \sigma(\mathbf{w}_i^T \mathbf{f} + b_i) = \frac{1}{1 + \exp(-(\mathbf{w}_i^T \mathbf{f} + b_i))}$$

2. **領藥天數自迴歸模型 (Ridge Auto-Regressive Model)**：
   僅選取歷史間隔天數序列 $\mathbf{d}_{\text{past}} = [d_1, d_2, \dots, d_T]$，擬合 Ridge L2 正則化迴歸：
   $$\hat{y}_g = \mathbf{w}_g^T \mathbf{d}_{\text{past}} + b_g$$
   $$\mathcal{L}_{\text{Ridge}} = \sum_{k=1}^N (y_{g, k} - \hat{y}_{g, k})^2 + \alpha \|\mathbf{w}_g\|_2^2, \quad (\alpha = 1.0)$$

---

## 3. 藥局零售連帶商品推薦模型 (Retail OTC Recommendation Models)

### 3.1 表格去噪擴散概率模型 (Tabular DDPM)

Tabular DDPM 用於學習高維度購物籃非處方商品 (OTC) 的條件分佈 $p(\mathbf{y}_{\text{retail}} \mid a, c)$。

#### (A) 表格目標連續化 (Target Rescaling)
將二元購物籃商品向量 $\mathbf{y}_{\text{retail}} \in \{0, 1\}^K$ ($K=10$) 映射至連續空間 $\mathbf{y}_0 \in \{-1, 1\}^K$：
$$\mathbf{y}_0 = 2 \mathbf{y}_{\text{retail}} - 1$$

#### (B) 正向加噪過程 (Forward Diffusion Process)
在時間步 $t \in \{1, 2, \dots, N_{\text{steps}}\}$（$N_{\text{steps}}=50$），依據預設的高斯方差排程 $\beta_t \in [0.0001, 0.02]$ 逐步添噪：
$$q(\mathbf{y}_t \mid \mathbf{y}_0) = \mathcal{N}\left(\mathbf{y}_t; \sqrt{\bar{\alpha}_t} \mathbf{y}_0, (1 - \bar{\alpha}_t) \mathbf{I}\right)$$
$$\mathbf{y}_t = \sqrt{\bar{\alpha}_t} \mathbf{y}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$
其中 $\alpha_t = 1 - \beta_t$，累積乘積 $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$。

#### (C) 條件去噪網絡 (Conditional Denoising Network)
去噪網絡 $\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{y}_t, a, c, t)$ 接收加噪樣本 $\mathbf{y}_t$、年齡 $a$、疾病類別 $c$ 與時間步 $t$：
1. **上下文融合嵌入 (Context Embedding)**：
   $$\mathbf{c}_{\text{emb}} = \text{LayerNorm}\left( \text{SiLU}\left( \mathbf{W}_c [\mathbf{e}_{\text{age}}(a) ; \mathbf{e}_{\text{disease}}(c)] + \mathbf{b}_c \right) \right)$$
2. **時間步嵌入 (Time Embedding)**：
   $$\mathbf{t}_{\text{emb}} = \text{MLP}_t(t / N_{\text{steps}})$$
3. **去噪 MLP 輸出**：
   $$\hat{\boldsymbol{\epsilon}}_\theta = \text{MLP}\left( [\mathbf{y}_t ; \mathbf{t}_{\text{emb}} ; \mathbf{c}_{\text{emb}}] \right)$$

#### (D) 訓練目標 (Loss Function)
去噪網絡學習預測添加的高斯噪聲 $\boldsymbol{\epsilon}$：
$$\mathcal{L}_{\text{DDPM}}(\theta) = \mathbb{E}_{t, \mathbf{y}_0, \boldsymbol{\epsilon}} \left[ \left\| \boldsymbol{\epsilon} - \hat{\boldsymbol{\epsilon}}_\theta(\mathbf{y}_t, a, c, t) \right\|^2 \right]$$

#### (E) 反向去噪採樣 (Reverse Sampling)
從純高斯噪聲 $\mathbf{y}_{N_{\text{steps}}} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 開始，依序進行 $N_{\text{steps}}$ 步遞推去噪：
$$\mathbf{y}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{y}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \hat{\boldsymbol{\epsilon}}_\theta(\mathbf{y}_t, a, c, t) \right) + \sqrt{\beta_t} \mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$
採樣完成後，還原為 [0, 1] 區間之購買機率：
$$\hat{\mathbf{p}}_{\text{retail}} = \frac{\mathbf{y}_0^{\text{sampled}} + 1}{2}$$

---

### 3.2 零售推薦前饋神經網路模型 (ProductRecommenderNN)

- **輸入特徵向量**：$\mathbf{x}_{\text{retail}} = [a / 100 ; \mathbf{m}] \in \mathbb{R}^{1 + M}$
- **前向運算**：
  $$\mathbf{h}_1 = \text{ReLU}(\mathbf{W}_1 \mathbf{x}_{\text{retail}} + \mathbf{b}_1)$$
  $$\mathbf{h}_2 = \text{ReLU}(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2)$$
  $$\hat{\mathbf{y}}_{\text{retail}} = \sigma(\mathbf{W}_3 \mathbf{h}_2 + \mathbf{b}_3)$$
- **優化目標**：
  $$\mathcal{L}_{\text{NN}} = \mathcal{L}_{\text{BCE}}(\hat{\mathbf{y}}_{\text{retail}}, \mathbf{y}_{\text{retail}})$$

---

## 4. 系統模型評估指標數學定義 (Evaluation Metrics)

設預測的二元商品/藥物集合為 $\mathcal{Y}_{\text{pred}}$，真實集合為 $\mathcal{Y}_{\text{true}}$：

### 4.1 精準率 (Precision)
$$\text{Precision} = \frac{|\mathcal{Y}_{\text{true}} \cap \mathcal{Y}_{\text{pred}}|}{|\mathcal{Y}_{\text{pred}}|}$$

### 4.2 召回率 (Recall)
$$\text{Recall} = \frac{|\mathcal{Y}_{\text{true}} \cap \mathcal{Y}_{\text{pred}}|}{|\mathcal{Y}_{\text{true}}|}$$

### 4.3 傑卡德相似度 / 交集並集比 (Jaccard Index / IoU)
$$\text{Jaccard Index} = \frac{|\mathcal{Y}_{\text{true}} \cap \mathcal{Y}_{\text{pred}}|}{|\mathcal{Y}_{\text{true}} \cup \mathcal{Y}_{\text{pred}}|}$$

對於連續批次數據，計算平均 Jaccard 相似度：
$$\text{IoU}_{\text{batch}} = \frac{1}{B} \sum_{b=1}^B \frac{\sum_{i=1}^M \left(\mathbb{I}(\sigma(\hat{z}_{b, i}) > 0.5) \cdot y_{b, i}\right)}{\sum_{i=1}^M \min\left(1, \mathbb{I}(\sigma(\hat{z}_{b, i}) > 0.5) + y_{b, i}\right) + \epsilon}$$
其中 $\epsilon = 10^{-8}$ 避免零除以零。
