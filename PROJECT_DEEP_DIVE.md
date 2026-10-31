# NEXUS — Complete Project Deep Dive

This document explains **everything** in the repository: the business problem, dataset, lakehouse pipeline, mathematics, models, optimization, forecasting, the production React + FastAPI product, each UI screen, and how the pieces connect. It is written so a teammate or examiner can understand the project without reading every notebook cell.

---

## 1. What NEXUS is

**NEXUS** is a customer behavioral intelligence product for eCommerce. It answers five business questions using the REES46 multi-category clickstream dataset (October 2019):

| # | Question | Method |
| --- | --- | --- |
| Q1 | Who are our customers? | Unsupervised segmentation (PCA + K-Means → 4 personas) |
| Q2 | Will they buy in the next 10 days? | Supervised purchase-intent model + Platt calibration |
| Q3 | Do segments really differ in conversion? | χ², pairwise z-tests, Wilson CIs, ANOVA/Tukey |
| Q4 | What will traffic look like next week? | Seasonal SARIMA forecast |
| Q5 | How should we allocate marketing budget? | Constrained Lagrange (KKT) study *and* product EV targeting |

The **product surface** is a React web app. A thin **FastAPI** layer wraps the same Python engines used in notebooks. Deep math lives under **Methodology** so operators see decisions first and examiners can still audit the science.

---

## 2. Repository map

```text
DSC_GP/
├── frontend/                 # Vite + React product UI
│   └── src/pages/            # Home, Customer, Campaign, Batch, Segments, Traffic, Methodology
├── backend/                  # FastAPI app
│   ├── main.py
│   └── routes/               # customers, batch, campaign, personas, forecast, methodology
├── src/                      # Core engines (shared by notebooks + API)
│   ├── scoring_engine.py     # Live scoring (scaler → PCA → KMeans → LR → Platt)
│   ├── campaign_optimizer.py # EV targeting + Lagrange solver
│   ├── pdf_generator.py      # Campaign PDF brief
│   ├── segmentation.py       # PCA / K-Means helpers
│   ├── optimization.py       # GD from scratch + profit curve
│   ├── budget_optimization.py
│   ├── timeseries.py
│   ├── stats.py
│   ├── etl.py, config.py, utils.py
├── models/                   # Serialized joblibs for production scoring
├── data/app_data/            # Small contracts served by the app (< few MB)
├── notebooks/                # Offline pipeline 01–07
├── reports/                  # CSV benchmarks + plots for Methodology
└── app/                      # Legacy Streamlit (retired; see app/LEGACY.md)
```

### How to run the product

```bash
# Terminal 1 — API
source .venv/bin/activate
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — UI
cd frontend && npm run dev
```

- UI: http://127.0.0.1:5173  
- API docs: http://127.0.0.1:8000/docs  

Vite proxies `/api/*` to port 8000.

---

## 3. REES46 dataset — what it is and how we broke it down

### 3.1 Source file

| Property | Value |
| --- | --- |
| Public name | REES46 / “eCommerce behavior data from multi-category store” (Kaggle: `mkechinov/...`) |
| File used | `data/raw/2019-Oct.csv` (~5.5–5.67 GB) |
| Period | **1–31 October 2019** (UTC timestamps) |
| Grain | **One row = one clickstream action** by one user on one product in one session |
| Raw row count | **42,448,764** |

This is **event-level** data, not a customer table. Almost everything useful for NEXUS (RFM, personas, intent labels) is **aggregated later** from these events.

### 3.2 Raw column dictionary (exact schema)

Notebook 01 applies an **explicit Spark schema** (faster and safer than `inferSchema`):

| Column | Type | Role | Notes |
| --- | --- | --- | --- |
| `event_time` | string → timestamp | When | e.g. `"2019-10-01 00:00:00 UTC"` |
| `event_type` | string | What | `view`, `cart`, `remove_from_cart`, `purchase` |
| `product_id` | long | Product key | Structured ID |
| `category_id` | long | Internal category ID | Structured |
| `category_code` | string | Human hierarchy | Semi-structured: `"electronics.smartphone"`; **~50% null** expected |
| `brand` | string | Brand | Often null → later imputed `"unknown"` |
| `price` | double | USD price | Must be > 0 after cleaning |
| `user_id` | long | Customer key | Sampling unit |
| `user_session` | string | Session UUID | Grouping key for sessions |

Data-type view (viva language):

- **Structured:** `price`, IDs  
- **Semi-structured:** `category_code` (dot hierarchy)  
- **Temporal:** `event_time`  
- **Categorical:** `event_type`, `brand`

### 3.3 Why we cannot “just open it in pandas”

42M rows × wide strings ≈ multi‑GB in memory; a typical **8 GB M1** laptop cannot hold the full CSV as a pandas DataFrame (~12 GB class footprint). Notebook 01 therefore:

1. Starts **PySpark local[*]** with driver memory capped at **4g**, shuffle partitions **8** (match cores).  
2. Requires **Java 17** (Java 22 breaks Spark reflection APIs).  
3. Reads lazily with the fixed schema, then runs **actions** only when counting / sampling / writing.

### 3.4 Deterministic 5% **user** sample (not row sample)

**Wrong approach:** keep 5% of *rows*. That shreds user timelines (user A’s 100 events might become 5 random crumbs) and destroys sessions, funnels, and the Oct 1–20 / 21–31 split.

**Correct approach (Bronze):**

1. Collect all distinct `user_id`.  
2. Hash each ID (MD5 → integer), take \(\mathrm{hash}(u) \bmod 100\).  
3. Keep user iff \(\mathrm{hash}(u) \bmod 100 < 5\) → **~5% of users**.  
4. Filter the **full event log** to those users only → each kept user retains **100% of their events**.

Config (`src/config.py`): `USER_SAMPLE_FRACTION = 0.05`, `USER_HASH_MOD = 100`.

Why MD5 instead of `user_id % 100`? Sequential IDs can bias modulo buckets; a cryptographic hash scatters IDs uniformly.

**Resulting Bronze size (from reconciliation):** **2,100,838** events (~5% of raw by user, not exactly 5% of rows).

Bronze is written as **Snappy Parquet** partitioned for fast column pruning (CSV → Parquet is also Unit 2 evidence: smaller disk, faster scans).

### 3.5 Quality checks on the sample (Phase 1)

Typical DQ performed before trusting Bronze:

- Null rates (especially `category_code`, `brand`)  
- Non-positive / absurd `price`  
- Duplicate exact rows  
- Session–user integrity (`user_session` belonging to one `user_id`)  
- Event-type distribution  

Outputs land under `reports/` (e.g. `data_quality_report_raw.csv`).

### 3.6 Reconciliation funnel (how 42M becomes ~111k customers)

From `reports/reconciliation_phase2.csv`:

| Stage | Rows / entities | What happened |
| --- | --- | --- |
| `01_raw_csv` | **42,448,764** events | Full October CSV |
| `02_bronze_sample` | **2,100,838** events | 5% user sample, histories intact |
| `03_silver_clean` | **2,096,110** events | Drop ~0.23%: `price ≤ 0`, exact duplicates; enrich categories |
| `04_gold_users` | **110,956** **customers** | One row per user active in feature window + label |

So: **events → cleaned events → one behavioral profile per customer**. The Gold mart is the modeling unit of analysis.

### 3.7 Temporal windows (the most important modeling decision)

| Window | Dates (inclusive) | Allowed use |
| --- | --- | --- |
| **Feature window** | **2019-10-01 → 2019-10-20** | All \(X\) signals (RFM, funnels, breadth) |
| **Label window** | **2019-10-21 → 2019-10-31** | Only the target \(y\) |

Cutoff is end of **20 Oct**. Any purchase on/after 21 Oct is **future** relative to features.

\[
y_i =
\begin{cases}
1 & \text{if user } i \text{ has } \ge 1 \text{ purchase with } \texttt{event\_date} \in [21,31]\ \text{Oct} \\
0 & \text{otherwise}
\end{cases}
\]

**Leakage example (what we forbid):** using “total purchases in all of October” as a feature to predict “bought in last 10 days” — that feature already contains the answer.

Positive rate on the Gold mart ≈ **3.87%** (severe imbalance). App warehouse (10k scored sample) ≈ **4.02%**.

### 3.8 App vs full Gold

| Artifact | Size | Role |
| --- | --- | --- |
| `data/gold/user_features.parquet` | ~110,956 users | Full training / stats mart |
| `data/app_data/scored_customers_warehouse.parquet` | **10,000 × 36** | Product browse / campaign pool / map |
| `models/*.joblib` | — | Fit on Gold; scored at runtime |

---

## 4. Medallion breakdown — Bronze → Silver → Gold (exact transforms)

```text
RAW 2019-Oct.csv (42.4M events)
        │  PySpark: schema + 5% user hash sample + DQ
        ▼
BRONZE  data/bronze/events_oct/*.parquet  (~2.10M events)
        │  DuckDB SQL in src/etl.build_silver
        ▼
SILVER  data/silver/events_oct/*.parquet  (~2.096M events)
        │  DuckDB star schema + gold_user_features SQL
        ▼
GOLD    warehouse.duckdb + user_features.parquet (~110,956 users)
        │  Notebook 04 adds cluster; 05 adds intent model
        ▼
APP     models/ + data/app_data/ contracts for React/API
```

### 4.1 Silver cleaning rules (`src/etl.py` → `build_silver`)

Applied in one DuckDB `SELECT DISTINCT ... WHERE price > 0`:

1. **Drop invalid prices:** keep only `price > 0`.  
2. **Deduplicate** exact duplicate event rows (`SELECT DISTINCT`).  
3. **Session key:** `session_id = concat(user_id, '_', user_session)` so sessions cannot collide across users.  
4. **Parse category hierarchy:**  
   - `"electronics.smartphone"` → `category_level_1 = electronics`, `category_level_2 = smartphone`  
   - Missing / empty → `"unknown"`  
5. **Impute brand:** `COALESCE(brand, 'unknown')`.  
6. **Temporal attributes:** `hour_of_day`, `day_of_week`, `is_weekend` (weekend if day ∈ {0,6} in DuckDB’s convention).  
7. Write Snappy Parquet **partitioned by `event_date`**.

### 4.2 Star schema in DuckDB (`build_star_schema`)

| Table | Grain | Contents |
| --- | --- | --- |
| `fact_events` | 1 row per cleaned event | FKs: `date_key`, `user_id`, `product_id`; degenerate: `session_id`, `event_type`; measure: `price` |
| `dim_date` | 1 row per calendar day in sample | `day_name`, weekend flag, week of year |
| `dim_product` | 1 row per `product_id` | Category L1/L2, brand, avg/median/min/max price, view/cart/purchase counts |
| `dim_user` | 1 row per `user_id` | First/last event, active days, sessions, lifetime spend (over Silver) |

Star schema = denormalized dimensions around a fact for fast BI aggregations (vs deep 3NF).

### 4.3 Gold feature mart SQL — how each feature is computed

Implemented in `build_gold_features` (DuckDB). Two CTEs:

**A. `feat_window`** — aggregate Silver where `event_date <= 2019-10-20`, `GROUP BY user_id`:

| Feature | Exact construction |
| --- | --- |
| `recency_days` | `datediff(day, max(event_date), 2019-10-20)` — days idle before cutoff |
| `tenure_days` | `datediff(day, min(event_date), 2019-10-20)` — how long observed |
| `active_days` | `count(distinct event_date)` |
| `total_sessions` | `count(distinct session_id)` |
| `total_events` | `count(*)` |
| `view_count` | `count(*) FILTER (event_type = 'view')` |
| `cart_count` | `count(*) FILTER (event_type = 'cart')` |
| `purchase_count` | `count(*) FILTER (event_type = 'purchase')` |
| `total_spend` | `sum(price) FILTER (purchase)` |
| `avg_purchase_price` | `avg(price) FILTER (purchase)` |
| `max_purchase_price` | `max(price) FILTER (purchase)` |
| `max_price_interacted` | `max(price)` over all event types |
| `avg_price_interacted` | `avg(price)` over all events |
| `n_unique_products` | `count(distinct product_id)` |
| `n_unique_cat1` | `count(distinct category_level_1)` |
| `n_unique_brands` | `count(distinct brand)` |
| `weekend_event_ratio` | weekend events / total events |
| `cart_to_view_ratio` | carts / views (0 if no views) |
| `purchase_to_cart_ratio` | purchases / carts |
| `purchase_to_view_ratio` | purchases / views |
| `events_per_session` | total events / distinct sessions |

**B. `label_window`** — Silver where `event_date >= 2019-10-21` **and** `event_type = 'purchase'`:

- `future_purchases`, `future_spend`  
- Left-join onto features → `will_purchase_next_10d = 1` iff `future_purchases > 0`.

Users with **no** feature-window activity never appear in Gold (they have nothing to score).

### 4.4 Notebook map

| Notebook | Layer / math | Deliverable |
| --- | --- | --- |
| 01 | Bronze + DQ | Sample Parquet, quality report |
| 02 | Silver + star + Gold features | `user_features.parquet`, DuckDB |
| 03 | EDA + hypothesis tests | Summary stats, H1–H3 |
| 04 | PCA / SVD / K-Means | Clusters + personas |
| 05 | Supervised + GD + τ* | Champion + calibrator joblibs |
| 06 | χ² + Lagrange | Segment tests + budget study |
| 07 | SARIMA | `forecast_7d.json` |

---

## 5. Feature space used for segmentation & scoring

### 5.1 The 17 numeric features for clustering / PCA

Same list in notebook 04 and `scoring_engine.NUMERIC_FEATURES`:

1. RFM-ish: `recency_days`, `tenure_days`, `active_days`, `total_sessions`, `total_events`  
2. Funnel volumes: `view_count`, `cart_count`, `purchase_count`  
3. Money: `total_spend`, `avg_purchase_price`, `max_price_interacted`  
4. Breadth: `n_unique_products`, `n_unique_cat1`, `n_unique_brands`  
5. Intensities: `weekend_event_ratio`, `cart_to_view_ratio`, `events_per_session`

### 5.2 Preprocessing before any linear algebra

For each user row (training):

1. Take the 17 columns.  
2. On **skewed** columns (spend, counts, prices, unique products, sessions):  
   \[
   x' = \log(1 + \max(0, x))
   \]  
   (`np.log1p`) — compresses heavy right tails so a few mega-browsers don’t dominate Euclidean distance.  
3. Fill remaining NaNs with 0.  
4. **Standardize:**
   \[
   z_{ij} = \frac{x'_{ij} - \mu_j}{\sigma_j}
   \]  
   so each column has mean ≈ 0, variance ≈ 1 (`StandardScaler`). Design matrix \(X \in \mathbb{R}^{n \times 17}\) with \(n \approx 110{,}956\).

Without standardization, `total_events` would drown `weekend_event_ratio` in K-Means.

### 5.3 Extra features for the intent model only

After clustering, append one-hots `cluster_0 … cluster_3` → **21** supervised inputs (see §7). Segmentation itself uses only the 17 continuous behaviors.

---

## 6. Unsupervised segmentation — mathematics in full (Q1)

Goal: discover **structure** in the 17-D behavior space, then name four **personas** marketers understand.

### 6.1 Step-by-step pipeline

```text
Gold 17 features
  → log1p (skewed) + StandardScaler          → X ∈ R^{n×17}
  → Covariance / SVD / PCA verification       → theory check
  → PCA to 5 components                       → Z ∈ R^{n×5}  (~80% variance)
  → K-Means++ for k=2..7 metrics              → choose k=4
  → Final K-Means (k=4, n_init=15)            → labels ∈ {0,1,2,3}
  → Profile + name personas                   → persona_profiles.json
  → Save pca_5comp.joblib, kmeans_4clusters.joblib, scaler_17num.joblib
```

Code: `src/segmentation.py` + `notebooks/04_segmentation.py`.

---

### 6.2 Covariance matrix

With \(X\) already column-centered by the scaler:

\[
C = \frac{1}{n-1}\, X^\top X \in \mathbb{R}^{17 \times 17}.
\]

Properties we verify:

- **Symmetry:** \(\|C - C^\top\|_F = 0\)  
- **PSD:** \(v^\top C v \ge 0\) for all \(v\) (variances / covariances)

Empirical audit (`linear_algebra_verifications.csv`): symmetry error **0**.

Interpretation: \(C_{jk}\) is how feature \(j\) co-moves with feature \(k\) across customers (e.g. sessions with events).

---

### 6.3 SVD and the link to PCA eigenvalues

Thin SVD of the centered matrix:

\[
X = U \Sigma V^\top
\]

- \(U\): left singular vectors (directions in customer space)  
- \(\Sigma = \mathrm{diag}(\sigma_1 \ge \sigma_2 \ge \cdots)\): singular values  
- \(V\): right singular vectors (directions in **feature** space) = PCA loading directions  

Eigendecomposition of covariance:

\[
C v = \lambda v.
\]

**Fundamental identity** (verified numerically):

\[
\lambda_i = \frac{\sigma_i^2}{n-1}.
\]

Audit: max \(|\lambda_i - \sigma_i^2/(n-1)| \approx 10^{-12}\) → **PASSED**.

PCA explained variance for component \(i\):

\[
\text{explained}_i = \frac{\lambda_i}{\sum_{j=1}^{p} \lambda_j}, \quad p=17.
\]

Notebook reports top PCs capture on the order of **~80%** cumulative variance with **5** components (enough to cluster without keeping all 17 noisy axes).

---

### 6.4 QR and Cholesky (syllabus verification, not used at score time)

| Decomposition | Identity checked | Result |
| --- | --- | --- |
| QR on a 5k-row subset \(X_{\mathrm{sub}}=QR\) | \(\|Q^\top Q - I\|_F \approx 0\) | PASSED (0) |
| Cholesky on \(C_{\mathrm{reg}}=C+\varepsilon I\) | \(\|C_{\mathrm{reg}} - LL^\top\|_F \approx 0\) | PASSED (0) |

These prove the feature covariance is well-behaved (symmetric PSD after tiny ridge) — examiner evidence that the linear-algebra stack is sound.

---

### 6.5 PCA projection (what we actually cluster on)

Fit sklearn `PCA(n_components=5)` on \(X\):

\[
Z = X V_{:,1:5}
\]

\(Z \in \mathbb{R}^{n \times 5}\). Loadings = columns of \(V\) (how much each original feature contributes to each PC).

**Qualitative reading from notebook comments (typical for this mart):**

| PC | Behavioral theme | High loadings tend to involve |
| --- | --- | --- |
| PC1 | Engagement volume | `total_events`, `view_count`, sessions, unique products |
| PC2 | Monetary / conversion intensity | `total_spend`, `purchase_count`, `cart_count` |
| PC3 | Lifecycle vs freshness | `tenure_days` vs `recency_days` |

Why reduce before K-Means?

- Euclidean K-Means suffers in high-D (distance concentration).  
- Correlated funnel counts are collapsed into fewer orthogonal axes.  
- PC1–PC2 also drive the **2D map** in the Segments UI (`pca_x`, `pca_y` in the warehouse).

Artifact: `models/pca_5comp.joblib`.

---

### 6.6 K-Means mathematics

On points \(z_i \in \mathbb{R}^5\), choose \(k\) centroids \(\mu_1,\ldots,\mu_k\) to (locally) minimize **inertia** (within-cluster sum of squares):

\[
J = \sum_{i=1}^{n} \min_{k \in \{1,\ldots,K\}} \| z_i - \mu_k \|_2^2.
\]

Algorithm (Lloyd / sklearn):

1. **Init:** `k-means++` (spread-out starts; better than pure random).  
2. **Assign:** each point to nearest centroid (Euclidean).  
3. **Update:** centroid = mean of assigned points.  
4. Repeat until labels stabilize; `n_init=10` (search) / `15` (final) restarts, keep best \(J\).

#### Choosing \(k\) — three metrics (`evaluate_cluster_k`, \(k=2..7\))

| Metric | Formula / idea | Prefer |
| --- | --- | --- |
| **Inertia** | \(J\) above | Elbow: diminishing drops as \(k\) grows |
| **Silhouette** | For each point: \((b-a)/\max(a,b)\) where \(a\)=intra distance, \(b\)=nearest other cluster | Higher (closer to 1) |
| **Davies–Bouldin** | Avg ratio of within-scatter to between-centroid separation | Lower |

Empirical table (`clustering_metrics.csv`):

| k | Inertia | Silhouette | Davies–Bouldin |
| --- | --- | --- | --- |
| 2 | 1,004,798 | **0.447** | 1.216 |
| 3 | 758,399 | 0.408 | 1.105 |
| **4** | **649,579** | 0.310 | 1.156 |
| 5 | 558,175 | 0.325 | **1.060** |
| 6 | 495,795 | 0.326 | 1.113 |
| 7 | 447,082 | 0.331 | 1.085 |

**Why \(k=4\) anyway?** Pure max-silhouette would prefer \(k=2\) (basically “active vs inactive”). Business needs a **taxonomy** (Champions / Explorers / Window / Dormant). \(k=4\) sits on a reasonable elbow of inertia and yields **stable, nameable** RFM-like groups. This is an explicit product/science tradeoff — say it in the viva.

Final fit: `KMeans(k=4, init="k-means++", n_init=15)` → `kmeans_4clusters.joblib`.

---

### 6.7 From cluster IDs to persona names

After labels exist, `profile_customer_segments` aggregates medians/means per cluster (recency, events, spend, conversion). Naming rules (business heuristics on the summary row):

```text
if mean_spend > 150 OR conversion > 10%     → Champions (High-Value Spenders)
else if median_events ≥ 15 OR conversion ≥ 4% → Active Explorers (High Engagement)
else if median_recency ≥ 8                  → Dormant Bouncers (Cold / Inactive)
else                                        → Recent Window Shoppers (Warm Bouncers)
```

Cluster integer IDs `{0,1,2,3}` are then aligned to those names in `persona_profiles.json` (including 17-D centroids and 2-D PCA centers).

### 6.8 Final persona card (full Gold mart)

| ID | Persona | \(n\) | % | Conversion (Wilson 95%) | Mean spend | Typical signal |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Dormant Bouncers | 39,911 | 36.0% | 1.28% [1.17, 1.39] | ~$1 | High recency (~13d), ~1 session, ~2 events |
| 1 | Recent Window Shoppers | 36,245 | 32.7% | 2.00% [1.86, 2.15] | ~$0.80 | Fresh (~3d) but still shallow |
| 2 | Champions | 11,266 | 10.2% | **16.93%** [16.25, 17.63] | **~$627** | Multi-session, real money |
| 3 | Active Explorers | 23,534 | 21.2% | 4.90% [4.63, 5.18] | ~$0.36 | High events/views, little spend yet |

Champions convert ≈ **13.2×** Dormant.

### 6.9 Wilson score interval (why not Wald)

For conversion \(\hat{p}=x/n\) with small \(p\), the normal Wald interval \(\hat{p}\pm z\sqrt{\hat{p}(1-\hat{p})/n}\) misbehaves. **Wilson** (implemented in `src/stats.wilson_proportion_ci` via statsmodels) recenters the interval and stays inside \((0,1)\). Product UI shows rates; Methodology/JSON keeps Wilson bounds for examiners.

### 6.10 Scoring a new customer into a segment (runtime)

In `score_customer_vector`:

1. Build 17-vector → log1p → `scaler_17num` → \(z \in \mathbb{R}^{17}\).  
2. `pca_5comp.transform(z)` → \(z_{\mathrm{pca}} \in \mathbb{R}^{5}\).  
3. `kmeans.predict(z_{\mathrm{pca}})` → cluster id.  
4. Also compute Euclidean distance in **17-D standardized space** to each persona centroid \(c_k\):

\[
d_k = \| z - c_k \|_2
\]

Nearest \(c_k\) should match the K-Means label (sanity / explainability). Distances appear in science payloads, not the marketer face.

---

## 6A. How the REES breakdown and segmentation connect (one picture)

```text
42.4M click events (REES46 Oct)
   │ user-hash 5%
   ▼
2.10M Bronze events
   │ price>0, dedupe, parse category, session_id
   ▼
2.096M Silver events
   │ aggregate ONLY Oct 1–20 → X; purchases Oct 21–31 → y
   ▼
110,956 Gold user rows (17+ behaviors + label)
   │ log1p + scale → PCA(5) → K-Means(4)
   ▼
Persona ∈ {Dormant, Window, Champions, Explorers}
   │ + cluster one-hots → logistic + Platt
   ▼
calibrated P(buy in next 10d) → product UI / campaign EV
```

Every UI “segment” chip and every campaign filter ultimately traces back to **this** chain—not to hand-labeled CRM tags.

---

## 7. Supervised purchase intent (Q2)

### Label

\[
y_i = \mathbf{1}\{\text{user } i \text{ purchases in Oct 21–31}\}
\]

### Class imbalance

Positive rate ≈ **3.87%**. Accuracy is a bad primary metric. The bake-off optimizes / reports:

- **PR-AUC** (Average Precision) — primary for rare positives  
- **ROC-AUC**  
- **Brier score** — probability calibration quality  
- F1 at τ = 0.50 (for reference only)

### Model competition (`reports/intent_model_comparison.csv`)

| Model | PR-AUC | ROC-AUC | Brier | F1@0.50 | Fit time (s) |
| --- | --- | --- | --- | --- | --- |
| **Logistic Regression (champion)** | **0.2450** | **0.7961** | 0.1749 | 0.1873 | 0.26 |
| LightGBM | 0.2407 | 0.7873 | 0.1560 | 0.1957 | 0.93 |
| XGBoost | 0.2357 | 0.7930 | 0.1612 | 0.1985 | 0.41 |
| Random Forest | 0.2168 | 0.7854 | 0.1330 | 0.2055 | 1.07 |

**Champion = balanced logistic regression** — best PR-AUC, fast, fully interpretable coefficients. Saved as `intent_model_champion.joblib`.

Baseline PR-AUC for a useless model is roughly the positive rate (~0.039). Champion PR-AUC **0.245 ≈ 6.3×** that floor.

### Logistic model math

After scaling the 21-dimensional feature vector \(x\):

\[
\text{logit}(p) = \beta_0 + \sum_{j=1}^{21} \beta_j x_j, \qquad
p = \sigma(\text{logit}) = \frac{1}{1+e^{-\text{logit}}}
\]

Intercept \(\beta_0 \approx -0.5346\) (`model_coefficients.json`).

Odds ratio for feature \(j\): \(e^{\beta_j}\). Examples:

| Feature | Coef | Odds ratio | Intuition |
| --- | --- | --- | --- |
| `total_events` | +4.41 | ~82.2 | More activity ↑ intent (strong) |
| `view_count` | −3.75 | ~0.024 | After controlling for events, pure views can dilute |
| `recency_days` | −0.46 | ~0.63 | Staler users ↓ intent |
| `avg_purchase_price` | +0.32 | ~1.38 | Higher ticket affinity ↑ |

(Always interpret coefficients **conditional on other features** — multicollinearity among event counts is expected.)

### Platt calibration

Raw logistic probabilities are passed through a **Platt calibrator** (`calibrated_intent_model.joblib`) so reported probabilities better match empirical frequencies. The product shows **calibrated** purchase likelihood.

Scoring engine also returns raw vs calibrated for Methodology / science payloads.

### Loss functions & gradient descent (notebook 05)

From-scratch logistic training compares:

- Batch GD  
- Stochastic GD  
- Mini-batch GD  

Losses studied: **binary cross-entropy (log-loss)** and **Brier score**. Convergence traces live in `reports/optimization_convergence.csv` (Methodology → Calibration).

### Cost-sensitive threshold \(\tau^*\)

Default τ = 0.50 is arbitrary under imbalance. Expected profit model used in training:

\[
\text{Net}(\tau) = TP(\tau)\cdot R - \bigl(TP(\tau)+FP(\tau)\bigr)\cdot C
\]

with example economics \(R=\$50\) revenue per true buyer, \(C=\$2\) cost per contact/offer.

From `optimal_threshold_analysis.csv`, profit is maximized near:

| | τ* ≈ 0.50 | Notes |
| --- | --- | --- |
| Targeted users | 5,313 | On the evaluation fold |
| Precision | 11.0% | |
| Recall | **68.3%** | Captures most buyers |
| Expected net profit | **+$18,674** | Vs worse thresholds that over-contact |

The **product Campaign page** does not hard-threshold at τ*; it **ranks by expected value** and fills budget capacity (see §9). Threshold analysis remains examiner evidence in Methodology.

---

## 8. Segment statistics (Q3)

### Omnibus conversion difference

Segments × buy/not-buy contingency → χ² test of independence. Reported project narrative: χ² ≈ **6296**, \(p\) essentially 0, Cramér’s \(V\) ≈ **0.24** (moderate association). Pairwise conversion contrasts are all significant after Bonferroni / FDR (`segment_pairwise_comparisons.csv`).

Examples (conversion %):

| Pair | Diff (pp) | Reject H₀ (Bonferroni) |
| --- | --- | --- |
| Dormant vs Champions | −15.65 | Yes |
| Window vs Champions | −14.93 | Yes |
| Champions vs Explorers | +12.03 | Yes |

### Other hypothesis tests (`hypothesis_test_results.csv`)

| Hypothesis | Test | Result |
| --- | --- | --- |
| H1 Activity: buyers vs non-buyers | Welch t / Mann–Whitney | Reject H₀ (large effect) |
| H2 Category ⊥ purchase | χ² | Reject H₀ (tiny Cramér’s V ≈ 0.022) |
| H3 Spend across tenure cohorts | Kruskal–Wallis | Reject H₀ |

---

## 9. Marketing allocation — two different stories

This distinction matters for demos and viva.

### A) Product Campaign Plan (what the UI does)

**Story:** Under budget \(B\) and cost-per-contact \(c\), contact capacity is

\[
N = \left\lfloor \frac{B}{c} \right\rfloor.
\]

1. Filter warehouse by **goal** (all / lapsed / cart abandoners / champions).  
2. Score expected value per customer:

\[
EV_i = \hat{p}_i \times \text{persona\_avg\_spend}_{k(i)}
\]

(using campaign spend map: Dormant $1, Window $0.80, Champions $626.88, Explorers $28.50).

3. Sort by \(EV_i\) descending; take top \(N\).  
4. Report **predicted** buyers \(\sum \hat{p}_i\), predicted revenue \(\sum EV_i\), predicted ROI.  
5. Export CSV + PDF brief.

**Disclaimer (shown in UI):** these are observational expectations from historical behavior, **not** measured causal lift from messaging.

### B) Lagrange budget study (Methodology only)

**Story:** Allocate continuous dollars \(b_i\) across 4 segments to maximize concave revenue with diminishing returns:

\[
\max_{b} \sum_{i=0}^{3} \alpha_i \ln(1 + \beta_i b_i)
\quad\text{s.t.}\quad
\sum_i b_i = B,\quad b_i \ge 0.
\]

Lagrangian / KKT first-order condition equalizes marginal returns to shadow price \(\lambda^*\):

\[
\frac{\partial R_i}{\partial b_i} = \frac{\alpha_i \beta_i}{1 + \beta_i b_i} = \lambda^*
\quad\Rightarrow\quad
b_i^*(\lambda) = \max\Bigl(0,\ \frac{\alpha_i}{\lambda} - \frac{1}{\beta_i}\Bigr).
\]

Solve for \(\lambda^*\) so \(\sum b_i^*(\lambda^*) = B\) (Brent root find in `campaign_optimizer.solve_lagrange_allocation`).

Example parameters (`LAGRANGE_PARAMS`):

| Segment | α | β |
| --- | --- | --- |
| Dormant | 4,200 | 0.00045 |
| Window | 9,500 | 0.00095 |
| Champions | 68,000 | 0.00280 |
| Explorers | 32,000 | 0.00160 |

At \(B=\$5{,}000\) in the live API, typical result: \(\lambda^* \approx 16.72\), most budget to Champions, remainder to Explorers, ~$0 to Dormant.

Benchmark at \(B=\$10{,}000\) (`lagrange_budget_allocation.csv`):

| Strategy | Net profit | vs Uniform |
| --- | --- | --- |
| **Lagrange optimal** | **$117,007** | Best |
| Uniform equal split | $99,393 | −15% |
| Size-proportional | $78,731 | Worse |

**These two engines are intentionally separate.** The pie/Lagrange study does **not** choose the product target list.

---

## 10. Traffic forecasting (Q4)

### Models compared (`timeseries_forecast_evaluation.csv`)

| Model | RMSE | MAE | MAPE % |
| --- | --- | --- | --- |
| **SARIMA(1,1,1)×(1,0,0)₇** | **2976.7** | **2488.8** | **3.99** |
| ARIMA(1,1,1) | 7125.4 | 6079.9 | 9.84 |
| Seasonal baseline | 6664.3 | 6360.4 | 10.04 |
| Holt–Winters | 7926.4 | 7505.0 | 11.91 |

Seasonal order 7 captures **weekly** shopping cycles.

### App contract

`forecast_7d.json` holds:

- 31 days of October history (`total_events`, `active_users`, purchases, GMV)  
- 7-day forecast (`predicted_traffic`, 80%/95% intervals, recommended action)

**Honesty fix in the product:** historical chart series uses **active users**; forecast uses visitor-scale predictions. Mixing raw `total_events` (~60–80k) with forecast (~13–16k) previously looked like a traffic cliff. MAPE/RMSE sit in Methodology, not as hero KPIs on Traffic.

---

## 11. Production scoring pipeline (runtime)

Implemented in `src/scoring_engine.py` and called by FastAPI:

```text
Feature dict (17 nums)
    → log1p on skewed columns
    → StandardScaler (scaler_17num)
    → PCA (5) → KMeans → cluster / persona
    → append cluster one-hots → 21 features
    → StandardScaler (scaler) → Logistic champion → raw p
    → Platt calibrator → calibrated p
    → plain-language reasons + recommended action
```

Artifacts:

| File | Role |
| --- | --- |
| `scaler_17num.joblib` | Scale for PCA/KMeans space |
| `pca_5comp.joblib` | Dimensionality reduction |
| `kmeans_4clusters.joblib` | Persona assignment |
| `scaler.joblib` | Scale for supervised 21-D input |
| `intent_model_champion.joblib` | Logistic regression |
| `calibrated_intent_model.joblib` | Platt calibration wrapper |
| `model_coefficients.json` | Interpretable β / odds ratios |
| `persona_profiles.json` | Centroids + segment stats |

Batch scoring (`score_batch_dataframe`) vectorizes the same path and ranks by calibrated probability.

---

## 12. Backend API (FastAPI)

Base: `backend/main.py` with CORS for the Vite origin.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Liveness |
| GET | `/api/customers/browse` | Filterable warehouse list (segment, likelihood, recency, spend, cart, sort) |
| GET | `/api/customers/{id}` | Score one warehouse customer |
| POST | `/api/score` | Score a ≤5-field synthetic profile |
| POST | `/api/batch` | Upload CSV → scored rows |
| GET | `/api/batch/demo` | Demo batch |
| GET | `/api/batch/template` | Simple CSV template download |
| POST | `/api/campaign` | EV targeting plan JSON |
| POST | `/api/campaign/csv` | Target list export |
| POST | `/api/campaign/pdf` | Executive brief PDF |
| GET | `/api/personas` | Segment cards |
| GET | `/api/segments/map` | PCA scatter sample |
| GET | `/api/forecast` | Traffic outlook (product-shaped) |
| GET | `/api/methodology/*` | models, calibration, segments, optimization, forecast, pipeline |

Simple profile / simple CSV columns are **mapped server-side** into the 17-feature schema when advanced columns are missing.

---

## 13. Frontend architecture

| Piece | Tech |
| --- | --- |
| Bundler | Vite 5 |
| UI | React 19 + React Router |
| Charts | Recharts |
| Styling | Plain CSS (`index.css`) — light paper / ink / forest accent (not neon Streamlit) |

### Global chrome

- Top nav: Lookup · Campaign · Batch · Segments · Traffic · Methodology  
- Footer disclaimer: predicted outcomes, not causal lift  
- Brand: quiet **NEXUS** wordmark  

### Design rules (product)

- Operator language first (likelihood, segment, action)  
- Math / λ* / Wilson / SARIMA order → Methodology  
- Cards only for interactive results  
- Subtle page-enter motion only  

---

## 14. UI screens — purpose, inputs, outputs, use cases

### 14.1 Home `/`

**Purpose:** Orient the user and route them into a job.

**Inputs:** None.

**Outputs:** Brand headline (“Know who will buy—and who to contact”), primary CTAs to Browse customers / Plan a campaign, secondary paths to Batch, Segments, Traffic.

**Use case:** Demo opening; teammate lands and knows what the product does in 5 seconds.

---

### 14.2 Customer Lookup `/customer`

**Purpose:** Score **one** shopper and get a next-best action.

**Modes:**

1. **Browse customers (default)**  
   - Filters: segment, min likelihood %, active-within days, min spend, cart nonempty, sort.  
   - Table of matching warehouse shoppers (display names like “Shopper 5276” — no memorized IDs).  
   - Click a row → live score via `/api/customers/{id}`.

2. **Build a profile**  
   - Five knobs: recency, spend, cart, sessions, events.  
   - Server expands to 17 features → score.

**Outputs:**

- Purchase likelihood (%) + intent band  
- Segment chip  
- “Why” bullets (heuristic reasons from spend/cart/recency/breadth)  
- Recommended action (VIP / discount / remarketing / re-engage by persona)  
- Link to Methodology  

**Use case:** “Show me a high-intent Champion and what we should do.” Marketer filters Champions + likelihood ≥ 10%, clicks a row.

---

### 14.3 Campaign Plan `/campaign`

**Purpose:** Decide **who to contact** under a budget.

**Inputs:**

- Goal: maximize return / win-back lapsed / convert cart abandoners / reward champions  
- Budget ($)  
- Cost per contact ($)  

**Outputs:**

- Contacts funded, predicted buyers, predicted revenue / ROI  
- Explicit story text (EV rank + capacity)  
- Disclaimer (not causal lift)  
- Preview table of top targets  
- Download CSV / PDF brief  
- Link to Methodology → Optimization for Lagrange study  

**Use case:** “We have $5,000 at $1/email — who should get it if we want cart abandoners?”

---

### 14.4 Batch Score `/batch`

**Purpose:** Score a **list** and download a ranked targeting file.

**Inputs:**

- Upload CSV (simple template: `customer_id, recency_days, total_spend, cart_count, total_sessions, total_events`)  
- Or one-click **demo batch**  
- Optional advanced 17-feature CSV still accepted  

**Outputs:**

- Count scored, predicted buyers, mean likelihood, count ≥ 10% intent  
- Ranked table  
- Download scored CSV  

**Use case:** Ops uploads a CRM extract; marketing gets a priority list for the week.

---

### 14.5 Segments `/segments`

**Purpose:** Explain **who the customers are**.

**Inputs:** Primary + comparison persona selectors.

**Outputs:**

- Size, conversion %, avg spend, recommended action for each  
- Behavioral map (PCA scatter of warehouse sample; color = segment)  

**Use case:** Stakeholder asks “why Champions matter”; compare Champions vs Dormant side by side.

---

### 14.6 Traffic `/traffic`

**Purpose:** Choose **when** to push campaigns.

**Inputs:** None (fixed 7-day outlook from precomputed SARIMA).

**Outputs:**

- Headline best days (typically Friday/Saturday)  
- History (active users) + forecast visitors chart  
- Day-by-day schedule with actions  
- Methodology link for MAPE / model order  

**Use case:** “Don’t blast Monday — launch into the weekend peak.”

---

### 14.7 Methodology `/methodology` (examiners)

**Purpose:** Prove the science without polluting product pages.

**Tabs:**

| Tab | Content |
| --- | --- |
| Models | Bake-off table, coefficient excerpt |
| Calibration | τ* profit curve table, GD convergence sample |
| Segments | Clustering metrics, hypothesis tests |
| Optimization | Live Lagrange solve (λ*, allocations) + benchmark CSV |
| Forecast | SARIMA order, MAPE/RMSE, stationarity tests |
| Pipeline | Bronze → Silver → Gold narrative + quality excerpt |

**Use case:** Viva / grading — invigilator opens Methodology only.

---

## 15. End-to-end demo script (suggested)

1. **Home** — state the three operator jobs: who buys, who to contact, when to send.  
2. **Lookup** — filter Champions, click a shopper, read likelihood + action.  
3. **Campaign** — $5,000 / $1 CPA / maximize return → show predicted buyers + download PDF.  
4. **Batch** — load demo CSV → download ranked file.  
5. **Segments** — Champions vs Dormant conversion gap.  
6. **Traffic** — Fri/Sat launch window.  
7. **Methodology** — show PR-AUC table, then Optimization λ*, then pipeline. Emphasize product EV ≠ Lagrange pie.

---

## 16. What was deliberately removed from the product

| Removed / demoted | Why |
| --- | --- |
| Streamlit as primary UI | Cluttered, neon, syllabus-first |
| Human-vs-model game | Not wanted for the product demo |
| Memorized customer ID entry | Replaced with filterable browse |
| Lagrange pie on Campaign | Misleading twin of targeting; kept in Methodology |
| Primary-surface Platt / KKT / Wilson / SARIMA latex | Confuses operators |
| Lakehouse + GD scrubber as a main tab | Methodology only |

---

## 17. Important caveats (say these out loud)

1. **Observational, not causal.** High \(\hat{p}\) means “looks like past buyers,” not “messaging will cause a purchase.”  
2. **Temporal split** protects training validity; the live app serves a **frozen** October 2019 world (warehouse + forecast).  
3. **App warehouse is 10k**, full mart is ~111k — UI is a demo slice of Gold.  
4. **Campaign EV spend map** can differ slightly from persona `mean_spend` in JSON (Explorers especially); campaign uses the optimizer’s commercial map.  
5. **Simple profile / CSV expansion** is a pragmatic mapping into 17 features — not a substitute for full feature ETL from raw clickstream.  
6. **Legacy Streamlit** under `app/` is not the supported runtime.

---

## 18. Key quantitative cheatsheet

| Metric | Value |
| --- | --- |
| Raw events | 42,448,764 |
| Gold users | ~110,956 |
| App warehouse | 10,000 |
| Base buy rate | ~3.87% |
| Champion PR-AUC | 0.2450 |
| Champion ROC-AUC | 0.7961 |
| Champions conversion | 16.93% |
| Dormant conversion | 1.28% |
| τ* fold profit (example economics) | +$18,674 @ ~68% recall |
| SARIMA holdout MAPE | 3.99% |
| Lagrange @ $10k vs uniform | ~+$17.6k net profit lift in study table |

---

## 19. File index for viva “show me the code”

| Ask | Open |
| --- | --- |
| How is a customer scored? | `src/scoring_engine.py` |
| How does campaign pick people? | `src/campaign_optimizer.py` → `run_campaign_optimization` |
| Where is Lagrange solved? | Same file → `solve_lagrange_allocation` |
| API browse filters? | `backend/routes/customers.py` |
| UI lookup? | `frontend/src/pages/Customer.jsx` |
| Persona numbers? | `data/app_data/persona_profiles.json` |
| Model bake-off? | `reports/intent_model_comparison.csv` |
| Temporal split explanation? | `notebooks/02_elt_and_star_schema.py` header |
| GD / τ*? | `notebooks/05_intent_prediction.py` |
| χ² / Lagrange notebook? | `notebooks/06_segment_statistics_and_optimization.py` |
| Forecast notebook? | `notebooks/07_time_series_forecasting.py` |

---

## 20. One-paragraph thesis

NEXUS turns 42M REES46 clickstream events into a Gold feature mart with a strict Oct 1–20 / 21–31 temporal split, segments users into four statistically distinct personas via PCA+K-Means, predicts 10-day purchase intent with a Platt-calibrated logistic champion (PR-AUC 0.245), studies diminishing-returns budget allocation via Lagrange multipliers for examiners, and ships a marketer-facing React product—browse/score customers, plan EV-ranked campaigns under budget, batch-score CSVs, explore segments, and time sends with SARIMA—while keeping the syllabus mathematics behind a dedicated Methodology door.

---

*Generated for the DSC_GP / NEXUS capstone. Prefer this document + Methodology UI for explanations; prefer notebooks for full reproducible training code.*
