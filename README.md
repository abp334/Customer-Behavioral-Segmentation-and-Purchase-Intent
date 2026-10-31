# Customer Behavioral Segmentation & Purchase Intent Prediction
### End-to-End Enterprise Data Science Capstone Project

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PySpark](https://img.shields.io/badge/PySpark-4.2.0-orange.svg)](https://spark.apache.org/)
[![DuckDB](https://img.shields.io/badge/DuckDB-1.1.0-yellow.svg)](https://duckdb.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40.0-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Executive Summary

This repository houses the complete, end-to-end production data science capstone project addressing the dual challenges of **customer behavioral segmentation** and **real-time purchase intent prediction** on massive eCommerce clickstream telemetry.

Using the **REES46 eCommerce Behavior Dataset** (October 2019: **42,448,764 events**, 5.67 GB raw CSV), we implement a complete Medallion Lakehouse architecture (Bronze $\to$ Silver $\to$ Gold Star Schema) on Apple Silicon M1 (8 GB RAM) under strict memory and latency boundaries. We answer all 5 core business questions with mathematical derivations, statistical hypothesis testing, machine learning benchmarks, constrained optimization, and time series forecasting.

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                MEDALLION LAKEHOUSE DATA ARCHITECTURE                              │
├────────────────────┬────────────────────┬─────────────────────┬───────────────────────────────────┤
│    RAW LAYER       │    BRONZE LAYER    │    SILVER LAYER     │            GOLD LAYER             │
│  42.45M Rows CSV   │  2.10M Rows Parquet│  2.09M Clean Parquet│  DuckDB Star Schema + Feature Mart│
│  Size: 5.67 GB     │  Size: 100 MB      │  Size: 98 MB        │  dim_date, dim_product, dim_user, │
│  Scan: 6.5s Spark  │  31 Partitions     │  Cleaned Prices > 0 │  fact_events & 110,956 Users Mart │
└────────────────────┴────────────────────┴─────────────────────┴───────────────────────────────────┘
```

---

## Core Business Questions Addressed

| # | Business Question | Methodology & Syllabus Units | Key Quantitative Finding |
| :--- | :--- | :--- | :--- |
| **Q1** | **What behavioral segments exist?** | RFM Engineering, SVD / PCA (Units 4 & 5), K-Means Clustering ($k=4$) | Identified 4 personas: **Champions** (10.1%, $626.88 spend), **Active Explorers** (21.2%), **Recent Window Shoppers** (32.7%), and **Dormant Bouncers** (36.0%). |
| **Q2** | **Who will purchase in next 10 days?** | Temporal split (Oct 01–20 vs 21–31), Gradient Descent from scratch (Unit 7), Supervised ML benchmark (Unit 8), Cost-Sensitive Thresholding | Balanced Logistic Regression achieved **PR-AUC 0.2450** (6.3× baseline). At threshold $\tau=0.50$, captures 68.3% buyers yielding **+$18,674.00 net profit**. |
| **Q3** | **Do segments differ in conversion?** | $\chi^2$ Omnibus Test, Bonferroni-adjusted Pairwise $z$-tests, One-way ANOVA & Tukey HSD (Unit 3 & 6) | Conversion difference is statistically indisputable ($\chi^2 = 6,295.96, p < 10^{-300}$, Cramér's $V = 0.2382$). Champions convert at **16.93%** (13.2× bouncers). |
| **Q4** | **What will traffic look like?** | Stationarity Tests (ADF & KPSS), ACF/PACF Analysis, Holt-Winters, ARIMA, Seasonal SARIMA (Unit 9) | SARIMA(1, 1, 1) $\times$ (1, 0, 0)$_7$ achieved **3.99% holdout MAPE** (Day 1 error only 72 events). Strong weekly rhythm peaks Sundays (~72k events). |
| **Q5** | **How should budget be allocated?** | Diminishing-Returns Log Concave Revenue, Analytical Lagrange Multipliers & KKT Conditions (Unit 6) | Shadow price $\lambda^* = 3.64$. On a $10,000 budget: allocate **67.4% to Champions**, 28.9% to Explorers, 3.7% to Shoppers, and **$0 to Dormant**, yielding **$117,007.15 profit (+17.6% lift)**. |

---

## Project Structure & Architecture

```
DSC_GP/
├── app/
│   ├── main.py                               # Executive Streamlit Navigation Hub
│   ├── components/
│   │   └── styles.py                         # Bespoke SaaS CSS & Styling System
│   └── views/
│       ├── p1_home.py                        # P1: Executive Overview & Live Preview
│       ├── p2_human_vs_model.py              # P2: Gamified Benchmark on Held-Out Test Clickstream
│       ├── p3_customer_lookup.py             # P3: Single Customer Scorer & Batch CSV Engine
│       ├── p4_campaign_builder.py            # P4: Lagrange Multiplier Budget & PDF Brief
│       ├── p5_persona_explorer.py            # P5: 4-Persona Explorer & 2D PCA Map
│       ├── p6_traffic_outlook.py             # P6: 7-Day SARIMA Forecast & Schedule
│       └── p7_how_it_was_built.py            # P7: Lakehouse Pipeline, GD Replay & Syllabus
├── src/
│   ├── scoring_engine.py                     # Real-time Platt Calibration & Feature Attribution
│   ├── campaign_optimizer.py                 # Constrained Lagrange KKT Multiplier Solver
│   ├── pdf_generator.py                      # Pure-Python Executive Campaign Brief PDF Builder
│   ├── segmentation.py                       # PCA & K-Means Unsupervised Pipeline
│   ├── optimization.py                       # Gradient Descent Variants from Scratch
│   ├── budget_optimization.py                # Analytical Lagrange Formulations
│   ├── timeseries.py                         # SARIMA & Holt-Winters Forecasting
│   └── stats.py                              # Chi-Square, Wilson CIs & ANOVA Tests
├── data/
│   ├── app_data/                             # Certified Precomputed App Contracts
│   │   ├── scored_customers_warehouse.parquet# 10,000 Pre-Scored Customer Profiles
│   │   ├── test_user_timelines.json          # 20 Held-Out Real Test Clickstream Timelines
│   │   ├── persona_profiles.json             # Centroids & 95% Wilson Confidence Intervals
│   │   ├── forecast_7d.json                  # SARIMA Forecast & Prediction Interval Bands
│   │   ├── model_coefficients.json           # Log-Odds Weights & Scaler Means
│   │   └── sample_upload_template.csv        # Downloadable Template for Batch Scoring
│   ├── bronze/events_oct/                    # Parquet: Deterministic 5% user sample
│   ├── silver/events_oct/                    # Parquet: Cleaned, deduplicated, enriched
│   └── gold/
│       ├── warehouse.duckdb                  # DuckDB In-Process Star Schema Warehouse
│       ├── user_features.parquet             # Gold Feature Mart (110,956 users, 25 features)
│       └── user_features_segmented.parquet   # Feature Mart enriched with PCA + Cluster IDs
├── models/
│   ├── intent_model_champion.joblib          # Serialized Balanced Logistic Regression
│   ├── scaler.joblib                         # Serialized StandardScaler
│   └── feature_names.joblib                  # Production feature list
├── notebooks/
│   ├── 01_setup_and_data_quality.py          # Phase 1: Spark Big Data Scan & Bronze Lakehouse
│   ├── 02_elt_and_star_schema.py             # Phase 2: Silver Cleaning, DuckDB Warehouse & Gold Mart
│   ├── 03_eda_and_statistics.py              # Phase 3: Skewness, Distributions & Hypothesis Tests
│   ├── 04_segmentation.py                    # Phase 4: SVD/PCA Decompositions & K-Means Personas
│   ├── 05_intent_prediction.py               # Phase 5: GD from Scratch, Supervised ML & Thresholding
│   ├── 06_segment_statistics_and_optimization.py # Phase 6: Chi-Square Tests & Lagrange Optimization
│   └── 07_time_series_forecasting.py         # Phase 7: ADF/KPSS Stationarity & SARIMA Holdout Forecast
├── reports/
│   ├── *.csv                                 # 12 Audit CSVs (reconciliation, tests, metrics)
│   ├── phase1_quality_overview.png           # Visual: Data Quality, Null Rates & Ingestion
│   ├── phase2_features_overview.png          # Visual: Feature Correlation Matrix & Distributions
│   ├── phase3_statistical_analysis.png       # Visual: Spend Skewness, Bootstrap CIs & Q-Q Plots
│   ├── phase4_segmentation_pca.png           # Visual: Scree Plot, PCA Biplot & Radar Personas
│   ├── phase5_intent_prediction.png          # Visual: Optimization Loss, PR Curves & Profit Curves
│   ├── phase6_segment_optimization.png       # Visual: Conversion CIs & Lagrange Budget Allocations
│   └── phase7_timeseries_forecasting.png     # Visual: ACF/PACF, Decomposition & SARIMA Holdout
├── src/
│   ├── config.py                             # Central configuration, file paths, seed definitions
│   ├── utils.py                              # Spark session builder, plotting helpers, formatters
│   ├── etl.py                                # Silver cleansing & DuckDB Star Schema builders
│   ├── stats.py                              # Confidence intervals, hypothesis testing, distributions
│   ├── segmentation.py                       # Covariance SVD, QR, Cholesky & K-Means clustering
│   ├── optimization.py                       # First-principles BGD/SGD/Mini-batch GD & ML benchmark
│   ├── budget_optimization.py                # Analytical Lagrange Multipliers & KKT solver
│   └── timeseries.py                         # Stationarity diagnostics, ARIMA & SARIMA forecasting
├── requirements.txt                          # Pinned dependency manifest
└── README.md                                 # Master documentation hub
```

---

## Detailed Pipeline & Phase Summaries

### Phase 1: Environment Setup & Big Data Quality Audit
- **Dataset Scale**: 42,448,764 clickstream events across 9 columns (5.67 GB).
- **Scan Performance**: PySpark 4.2.0 multi-threaded scan executed in **6.5 seconds** on Apple Silicon M1.
- **Deterministic Sampling Strategy**: MD5 hash modulo (`user_id % 100 < 5`), preserving 100% of event sequences for sampled users (random row sampling was rejected because it fragments conversion funnels).
- **Bronze Lakehouse**: 2,100,838 rows (151,177 users) written to Snappy Parquet partitioned by 31 dates (100 MB disk footprint, **98.1% storage reduction**, 10.1× scan acceleration).

### Phase 2: ELT Pipeline, DuckDB Star Schema & Feature Mart
- **Silver Cleansing**: Dropped 4,728 non-positive prices and duplicate timestamps; parsed `category_code` into a 3-tier hierarchy (`cat_level1`, `cat_level2`, `cat_level3`).
- **Star Schema (`warehouse.duckdb`)**:
  - `dim_date`: 31 records (day of week, weekend flags).
  - `dim_product`: 93,941 unique products with historical price statistics.
  - `dim_user`: 151,133 unique customer records.
  - `fact_events`: 2,096,110 cleaned events referencing dimension surrogates.
- **Gold Feature Mart (`user_features.parquet`)**: 110,956 users across 25 engineered features. Strict temporal windowing (Oct 01–20 features vs Oct 21–31 labels) guarantees **zero lookahead leakage**. Conversion rate: **3.87%** (4,292 buyers vs 106,664 non-buyers).

### Phase 3: Exploratory Data Analysis & Statistical Inference
- **Distribution Diagnostics**: Customer spend exhibits extreme positive skewness ($\gamma_1 = 22.77$) and heavy kurtosis ($\gamma_2 = 936.85$), proving $\log(1+x)$ transformations are mandatory.
- **Session Dispersion**: $\text{Variance}/\text{Mean} = 5.13 \gg 1.0$, refuting Poisson assumptions and validating Negative Binomial overdispersion.
- **Distribution Fitting**: Spend fits Log-Normal ($D=0.0290, \text{AIC}=176,680$) and Pareto, decisively outperforming Normal ($D=0.4283$).
- **Hypothesis Testing**: Welch's $t$-test ($t=26.22, p < 10^{-140}$), Mann-Whitney $U$ ($p < 10^{-300}$), and $\chi^2$ association ($p < 10^{-9}$) all reject the null at machine precision.

### Phase 4: Linear Algebra & Customer Segmentation (Q1)
- **Matrix Decompositions**: Verified covariance symmetry ($\|C - C^T\|_\infty = 0.00$), SVD singular values vs sample covariance eigenvalues ($\max |\lambda_i - \sigma_i^2/(n-1)| = 1.00 \times 10^{-12}$), QR orthogonality ($\|Q^T Q - I\|_\infty = 0.00$), and Cholesky reconstruction ($\|L L^T - C\|_\infty = 0.00$).
- **PCA**: Top 5 principal components capture **80.37% cumulative variance** (PC1: Engagement Volume 40.5%, PC2: Monetary Spend 17.1%, PC3: Recency/Tenure 10.2%).
- **K-Means Clustering ($k=4$)**:
  - **Champions** (10.15%): $626.88 spend, 16.93% 10-day conversion rate.
  - **Active Explorers** (21.21%): 20 median events, 4.90% conversion rate.
  - **Recent Window Shoppers** (32.67%): 3-day recency, 2.00% conversion rate.
  - **Dormant Bouncers** (35.97%): 13-day recency, 1.28% conversion rate.

### Phase 5: First-Principles Optimization & ML Benchmark (Q2)
- **Gradient Descent from Scratch**: Derived analytical binary cross-entropy gradient $\nabla_w J(w) = \frac{1}{N} X^T (\hat{y} - y)$. Mini-Batch GD converged to 0.1320 log-loss in **0.06 seconds**.
- **Supervised Benchmark (Holdout $N = 22,192$)**:
  - Balanced Logistic Regression: **PR-AUC 0.2450**, ROC-AUC 0.7961 (Champion).
  - LightGBM: PR-AUC 0.2407, ROC-AUC 0.7873.
  - XGBoost: PR-AUC 0.2357, ROC-AUC 0.7884.
  - Random Forest: PR-AUC 0.2168, ROC-AUC 0.7725.
- **Cost-Sensitive Thresholding**: With Reward $R = \$50$ and Cost $C = \$2$, optimal decision threshold $\tau = 0.50$ captures 68.3% of true buyers, generating **+$18,674.00 net profit**.

### Phase 6: Segment Hypothesis Testing (Q3) & Budget Optimization (Q5)
- **Omnibus $\chi^2$ Test**: $\chi^2 = 6,295.96$ ($df=3, p < 10^{-300}$, Cramér's $V = 0.2382$). All pairwise $z$-tests reject $H_0$ under Bonferroni/FDR ($p < 10^{-14}$).
- **Lagrange Multiplier Allocation**: Solved analytical KKT system for concave diminishing-returns revenue $R(b) = \alpha \ln(1 + \beta b)$:
  $$\lambda^* = 3.6443 \quad (\text{Shadow Price: } \$3.64\text{ return per incremental dollar})$$
- On a $\$10,000$ budget: allocates **$\$6,735.12$ to Champions** (67.4%), $\$2,892.86$ to Explorers (28.9%), $\$372.02$ to Shoppers (3.7%), and **$\$0.00$ to Dormant Bouncers**, producing **$\$117,007.15$ net profit (+$17,614.54 lift over equal allocation)**.

### Phase 7: Time Series Analysis & Traffic Forecasting (Q4)
- **Stationarity**: ADF test on raw daily events ($p = 0.2135$) and KPSS test ($p = 0.0412$) confirmed non-stationarity. First-order differencing achieved stationarity ($p < 10^{-5}$).
- **Rhythm & Seasonality**: ACF lag 7 confirmed a weekly cycle. Sundays peak at $\sim 72,000$ events; Thursdays trough at $\sim 63,000$ events.
- **Holdout Evaluation**:
  - **Seasonal SARIMA(1, 1, 1) $\times$ (1, 0, 0)$_7$**: **MAPE 3.99%** (Holdout Day 1 error: 72 events / 0.11%).
  - Non-seasonal ARIMA(1, 1, 1): MAPE 9.84%.
  - Holt-Winters Exponential Smoothing: MAPE 11.91%.

### Phase 8: Production Web Application & BI Deployment
- **Interactive Streamlit Web App (`app/main.py`)**: 5 tabs supporting executive KPIs, persona explorations, real-time customer intent scoring ($< 4\text{ ms}$ latency), dynamic budget optimization sliders, and time series forecasts.
- **Enterprise BI Marts (`dashboard/`)**: Curated dimensional extracts and workbook blueprints for Tableau and Microsoft Power BI.

---

## Quickstart & Reproduction Guide

### 1. Prerequisites
- macOS (Apple Silicon M1/M2/M3) or Linux x86_64
- Python 3.10 virtual environment
- OpenJDK 17 (`brew install openjdk@17`)
- OpenMP runtime (`brew install libomp`)

### 2. Setup Environment
```bash
# Clone repository
git clone <repo-url> DSC_GP
cd DSC_GP

# Create and activate virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Execution Pipeline (Sequential Phases 1–7)
All notebooks are self-contained executable Python modules:

```bash
# Phase 1: Environment & Bronze Lakehouse Ingestion
.venv/bin/python3 notebooks/01_setup_and_data_quality.py

# Phase 2: Silver Cleansing, DuckDB Star Schema & Feature Mart
.venv/bin/python3 notebooks/02_elt_and_star_schema.py

# Phase 3: Exploratory Data Analysis & Statistical Inference
.venv/bin/python3 notebooks/03_eda_and_statistics.py

# Phase 4: Linear Algebra Decompositions & K-Means Segmentation
.venv/bin/python3 notebooks/04_segmentation.py

# Phase 5: Optimization from Scratch & Intent Prediction Benchmark
.venv/bin/python3 notebooks/05_intent_prediction.py

# Phase 6: Segment Hypothesis Testing & Lagrange Budget Optimization
.venv/bin/python3 notebooks/06_segment_statistics_and_optimization.py

# Phase 7: Time Series Modeling & SARIMA Traffic Forecasting
.venv/bin/python3 notebooks/07_time_series_forecasting.py
```

### 4. Launch Production Streamlit Application
```bash
.venv/bin/streamlit run app/main.py --server.port 8501
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Viva Voce & Technical Defense Cheatsheet

### 1. Big Data & System Constraints
- **Why use PySpark broadcast join for sampling?**  
  A Python UDF forces row-by-row socket serialization across the JVM and Python worker processes, causing memory pressure on an 8 GB system. A native JVM broadcast hash join (`F.broadcast(sampled_df)`) keeps execution in native bytecode, completing the 42M row scan in 28.4 seconds.
- **Why DuckDB instead of SQLite or PostgreSQL?**  
  DuckDB is a vectorized, columnar in-process analytical engine. Analytical aggregations (`SUM`, `AVG`, window functions) run directly on columnar chunks in SIMD registers with zero client-server serialization overhead.

### 2. Statistics & Linear Algebra
- **Why Wilson Score Interval over Wald Interval?**  
  The Wald interval ($\hat{p} \pm z \sqrt{\hat{p}(1-\hat{p})/n}$) assumes normal approximation of binomial proportions, which collapses near $p \to 0$ and can generate intervals extending below zero. The Wilson Score interval inverts the score test, guaranteeing valid bounds even under rare conversion rates ($3.87\%$).
- **How does SVD connect to Sample Covariance Eigenvalues?**  
  For centered data matrix $X \in \mathbb{R}^{n \times d}$, sample covariance is $C = \frac{1}{n-1} X^T X$. Substituting the thin SVD $X = U \Sigma V^T$:
  $$C = \frac{1}{n-1} (V \Sigma U^T)(U \Sigma V^T) = V \left( \frac{\Sigma^2}{n-1} \right) V^T$$
  Thus, right singular vectors $V$ are the principal eigenvectors of $C$, and eigenvalues $\lambda_i = \sigma_i^2 / (n-1)$, verified to machine precision ($10^{-12}$).

### 3. Machine Learning & Optimization
- **Why is PR-AUC prioritized over ROC-AUC for Intent Prediction?**  
  With a 3.87% positive conversion rate, non-buyers vastly outnumber buyers (27:1). ROC-AUC's False Positive Rate ($FP / (FP + TN)$) is dominated by the massive $TN$ denominator, giving an overly optimistic score. PR-AUC evaluates Precision ($TP / (TP + FP)$) directly against Recall ($TP / (TP + FN)$), reflecting true economic targeting efficiency.
- **Why does the Lagrange allocation give $0 to Dormant Bouncers?**  
  The Karush-Kuhn-Tucker (KKT) dual stationarity condition requires marginal revenue to equal shadow price $\lambda^*$:
  $$\frac{\alpha_i \beta_i}{1 + \beta_i b_i} = \lambda^* \implies b_i^* = \max\left(0, \frac{\alpha_i}{\lambda^*} - \frac{1}{\beta_i}\right)$$
  For Dormant Bouncers, initial marginal productivity $\alpha_0 \beta_0 = 4000 \times 0.0005 = 2.00$ is strictly less than the equilibrium shadow price $\lambda^* = 3.64$. Allocating even $\$1$ to Dormant Bouncers would yield a negative marginal return compared to investing that dollar in Champions ($\alpha_2 \beta_2 = 200.0$).

---

## Authors & Acknowledgments

- **Author**: Aayush Pandya
- **Capstone Project**: Master of Science in Data Science / Advanced Machine Learning
- **Dataset**: Provided by REES46 eCommerce Technologies (October 2019) via Kaggle.
