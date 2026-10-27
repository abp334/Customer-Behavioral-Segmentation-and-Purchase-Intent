"""
============================================================
NOTEBOOK 05 – OPTIMIZATION & PURCHASE INTENT PREDICTION
============================================================
Phase 5  |  Syllabus Units: 7 (Optimization & Loss Functions),
                           8 (Supervised Machine Learning & Evaluation)
============================================================

OBJECTIVE
---------
1. Implement and compare Loss Functions: Binary Cross-Entropy (Log-Loss) and Brier Score.
2. Build Gradient Descent algorithms from scratch:
   - Batch Gradient Descent (BGD)
   - Stochastic Gradient Descent (SGD)
   - Mini-Batch Gradient Descent (MBGD)
3. Benchmark four supervised ML models for purchase intent:
   - Logistic Regression (Balanced)
   - Random Forest (Balanced)
   - LightGBM Classifier (Scale-pos-weight)
   - XGBoost Classifier (Scale-pos-weight)
4. Address severe 3.87% class imbalance using PR-AUC, ROC-AUC, and Brier Score.
5. Perform Cost-Sensitive Threshold Optimization:
   Identify optimal business threshold tau* maximizing expected commercial profit.
6. Save best trained model artifact for production deployment in Streamlit app.
"""

import sys, os, time, warnings
warnings.filterwarnings("ignore")

if os.path.basename(os.getcwd()) == "notebooks":
    sys.path.insert(0, os.path.dirname(os.getcwd()))
else:
    sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_recall_curve, roc_curve, classification_report, confusion_matrix
)
import xgboost as xgb
import lightgbm as lgb

from src.config import (
    PROJECT_ROOT, GOLD_DIR, MODELS_DIR, REPORTS_DIR, RANDOM_SEED
)
from src.optimization import (
    LogisticRegressionFromScratch, binary_cross_entropy_loss,
    compute_expected_profit_curve
)

print("=" * 60)
print("PHASE 5 – OPTIMIZATION & PURCHASE INTENT PREDICTION")
print("=" * 60)

# ============================================================
# CELL 1 – Load Segmented Feature Mart & Train/Test Split
# ============================================================
segmented_path = GOLD_DIR / "user_features_segmented.parquet"
df_users = pd.read_parquet(segmented_path)
print(f"Loaded {len(df_users):,} customer profiles with cluster assignments")

# Feature definition
numeric_features = [
    "recency_days", "tenure_days", "active_days", "total_sessions",
    "total_events", "view_count", "cart_count", "purchase_count",
    "total_spend", "avg_purchase_price", "max_price_interacted",
    "n_unique_products", "n_unique_cat1", "n_unique_brands",
    "weekend_event_ratio", "cart_to_view_ratio", "events_per_session"
]
skewed_features = [
    "total_spend", "total_events", "view_count", "cart_count", "purchase_count",
    "total_sessions", "avg_purchase_price", "max_price_interacted", "n_unique_products"
]

# One-hot encode the 4 behavioral clusters from Phase 4
cluster_dummies = pd.get_dummies(df_users["cluster"], prefix="cluster", drop_first=False).astype(float)

# Prepare feature matrix X and target y
X_df = df_users[numeric_features].copy()
for col in skewed_features:
    X_df[col] = np.log1p(np.maximum(0, X_df[col].fillna(0)))

X_full = pd.concat([X_df, cluster_dummies], axis=1)
y_full = df_users["will_purchase_next_10d"].values
all_feature_names = X_full.columns.tolist()

# 80/20 Stratified Train/Test Split (Preserving 3.87% positive conversion rate)
X_train_df, X_test_df, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.20, random_state=RANDOM_SEED, stratify=y_full
)

print(f"\nTrain set shape : {X_train_df.shape} (Positive buyers: {np.sum(y_train):,} / {np.mean(y_train)*100:.2f}%)")
print(f"Test set shape  : {X_test_df.shape} (Positive buyers: {np.sum(y_test):,} / {np.mean(y_test)*100:.2f}%)")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_df)
X_test_scaled = scaler.transform(X_test_df)

# Save preprocessor artifact
MODELS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
joblib.dump(all_feature_names, MODELS_DIR / "feature_names.joblib")
print(f"✅ Scaler artifact saved: {MODELS_DIR / 'scaler.joblib'}")

# ============================================================
# CELL 2 – Gradient Descent from Scratch (Unit 7 Optimization)
# ============================================================
"""
SYLLABUS UNIT 7 (Optimization & Loss Functions):
We implement and compare 3 variations of Gradient Descent from scratch:
1. Batch Gradient Descent (BGD): Updates weights using all training samples.
   theta <- theta - alpha * (1/n) * X^T (sigma(X*theta) - y)
2. Stochastic Gradient Descent (SGD): Updates weights using 1 random sample at a time.
   theta <- theta - alpha_t * x_i * (sigma(x_i*theta) - y_i)
3. Mini-Batch Gradient Descent (MBGD): Updates weights using mini-batches (B = 256).
   theta <- theta - alpha * (1/B) * X_b^T (sigma(X_b*theta) - y_b)
"""
print("\n" + "=" * 60)
print("TRAINING GRADIENT DESCENT ALGORITHMS FROM SCRATCH")
print("=" * 60)

# Train on a representative training sample (15,000 users) for rapid convergence benchmarking
sample_n = 15000
rng_opt = np.random.default_rng(RANDOM_SEED)
opt_idx = rng_opt.choice(len(X_train_scaled), size=sample_n, replace=False)
X_opt = X_train_scaled[opt_idx]
y_opt = y_train[opt_idx]

n_epochs = 35

print("1. Running Batch Gradient Descent (BGD)...")
t0 = time.perf_counter()
model_bgd = LogisticRegressionFromScratch(learning_rate=0.20, n_epochs=n_epochs, method="batch", random_state=42)
model_bgd.fit(X_opt, y_opt)
bgd_time = time.perf_counter() - t0
print(f"   BGD finished in {bgd_time:.2f}s | Final Log-Loss: {model_bgd.loss_history[-1]:.4f}")

print("2. Running Stochastic Gradient Descent (SGD)...")
t0 = time.perf_counter()
model_sgd = LogisticRegressionFromScratch(learning_rate=0.01, n_epochs=n_epochs, method="sgd", random_state=42)
model_sgd.fit(X_opt, y_opt)
sgd_time = time.perf_counter() - t0
print(f"   SGD finished in {sgd_time:.2f}s | Final Log-Loss: {model_sgd.loss_history[-1]:.4f}")

print("3. Running Mini-Batch Gradient Descent (MBGD, Batch=256)...")
t0 = time.perf_counter()
model_mbgd = LogisticRegressionFromScratch(learning_rate=0.10, n_epochs=n_epochs, batch_size=256, method="mini_batch", random_state=42)
model_mbgd.fit(X_opt, y_opt)
mbgd_time = time.perf_counter() - t0
print(f"   MBGD finished in {mbgd_time:.2f}s | Final Log-Loss: {model_mbgd.loss_history[-1]:.4f}")

# Save loss histories
loss_df = pd.DataFrame({
    "epoch": np.arange(1, n_epochs + 1),
    "batch_gd_loss": model_bgd.loss_history,
    "sgd_loss": model_sgd.loss_history,
    "mini_batch_gd_loss": model_mbgd.loss_history
})
loss_df.to_csv(REPORTS_DIR / "optimization_convergence.csv", index=False)

# ============================================================
# CELL 3 – Supervised Model Training & Benchmarking
# ============================================================
"""
SYLLABUS UNIT 8 (Supervised Machine Learning & Evaluation):
We train 4 candidate models with class imbalance weighting:
- Logistic Regression (L2 penalized, balanced)
- Random Forest Classifier (Ensemble bagging, balanced)
- LightGBM Classifier (Gradient boosted decision trees, scale_pos_weight=24.8)
- XGBoost Classifier (Gradient boosted decision trees, scale_pos_weight=24.8)
"""
print("\n" + "=" * 60)
print("TRAINING SUPERVISED INTENT MODELS")
print("=" * 60)

scale_pos_weight = (len(y_train) - np.sum(y_train)) / np.sum(y_train)
print(f"Class imbalance ratio (scale_pos_weight): {scale_pos_weight:.2f}")

models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=500, random_state=RANDOM_SEED
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=10, class_weight="balanced",
        n_jobs=-1, random_state=RANDOM_SEED
    ),
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=150, learning_rate=0.05, max_depth=6, num_leaves=31,
        scale_pos_weight=scale_pos_weight, random_state=RANDOM_SEED, n_jobs=-1, verbose=-1
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=150, learning_rate=0.05, max_depth=5,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss",
        random_state=RANDOM_SEED, n_jobs=-1
    )
}

results = []
test_predictions = {}

for name, clf in models.items():
    t0 = time.perf_counter()
    clf.fit(X_train_scaled, y_train)
    fit_time = time.perf_counter() - t0

    # Probabilistic prediction on test set
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]
    test_predictions[name] = y_prob

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    
    # Calculate F1 score at default threshold 0.50
    y_pred_default = (y_prob >= 0.50).astype(int)
    f1_default = 2 * np.sum((y_pred_default == 1) & (y_test == 1)) / (np.sum(y_pred_default == 1) + np.sum(y_test == 1))

    results.append({
        "model": name,
        "pr_auc": round(float(pr_auc), 4),
        "roc_auc": round(float(roc_auc), 4),
        "brier_score": round(float(brier), 4),
        "f1_score_tau50": round(float(f1_default), 4),
        "fit_time_seconds": round(float(fit_time), 2)
    })
    print(f"✅ {name:<20} | PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f} | Brier: {brier:.4f} | Train Time: {fit_time:.2f}s")

models_benchmark_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
models_benchmark_df.to_csv(REPORTS_DIR / "intent_model_comparison.csv", index=False)

print("\n" + "=" * 60)
print("SUPERVISED MODEL BENCHMARK RESULTS")
print("=" * 60)
print(models_benchmark_df.to_string(index=False))

# Identify best model by PR-AUC
best_model_name = models_benchmark_df.iloc[0]["model"]
best_model = models[best_model_name]
best_probs = test_predictions[best_model_name]
print(f"\n🏆 Champion Model: {best_model_name}")

# ============================================================
# CELL 4 – Cost-Sensitive Threshold Optimization (Question Q2)
# ============================================================
"""
COMMERCIAL THRESHOLD OPTIMIZATION:
Default threshold tau = 0.50 is arbitrary and disastrous for imbalanced commercial conversion.
We model real eCommerce unit economics:
- Expected Revenue per converted buyer : R = $50.00
- Cost per targeted promotional offer : C = $2.00
Expected Net Profit(tau) = TP(tau) * R - (TP + FP) * C
We search tau in [0.01, 0.90] to find the profit-maximizing threshold tau*.
"""
print("\n" + "=" * 60)
print("COST-SENSITIVE THRESHOLD OPTIMIZATION")
print("=" * 60)

profit_df = compute_expected_profit_curve(
    y_test, best_probs, revenue_per_buyer=50.0, cost_per_offer=2.0, n_thresholds=100
)
profit_df.to_csv(REPORTS_DIR / "optimal_threshold_analysis.csv", index=False)

# Optimal threshold row
max_profit_row = profit_df.loc[profit_df["expected_net_profit"].idxmax()]
default_tau_row = profit_df.iloc[(profit_df["threshold"] - 0.50).abs().argsort()[:1]].iloc[0]

opt_tau = max_profit_row["threshold"]
opt_profit = max_profit_row["expected_net_profit"]
opt_f1 = max_profit_row["f1_score"]
opt_targeted = max_profit_row["targeted_users"]
opt_precision = max_profit_row["precision"]
opt_recall = max_profit_row["recall"]

print(f"Baseline Naive Threshold (tau = 0.50):")
print(f"  Targeted Users : {int(default_tau_row['targeted_users']):,}")
print(f"  Net Profit     : ${default_tau_row['expected_net_profit']:,.2f}")
print(f"  Precision      : {default_tau_row['precision']*100:.1f}% | Recall: {default_tau_row['recall']*100:.1f}%")

print(f"\nOptimal Commercial Threshold (tau* = {opt_tau:.2f}):")
print(f"  Targeted Users : {opt_targeted:,}")
print(f"  Net Profit     : ${opt_profit:,.2f}  (Profit Gain: +${opt_profit - default_tau_row['expected_net_profit']:,.2f}!)")
print(f"  Precision      : {opt_precision*100:.1f}% | Recall: {opt_recall*100:.1f}% | F1-Score: {opt_f1:.4f}")

# ============================================================
# CELL 5 – Feature Importance & Interpretability
# ============================================================
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE & EXPLAINABILITY")
print("=" * 60)

if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    feat_imp_df = pd.DataFrame({
        "feature": all_feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)
    print("Top 10 Drivers of Purchase Intent:")
    print(feat_imp_df.head(10).to_string(index=False))
else:
    coefs = best_model.coef_[0]
    feat_imp_df = pd.DataFrame({
        "feature": all_feature_names,
        "coefficient": coefs
    }).sort_values("coefficient", key=abs, ascending=False)
    print("Top 10 Coefficients:")
    print(feat_imp_df.head(10).to_string(index=False))

# ============================================================
# CELL 6 – Multi-Panel Visualization Dashboard
# ============================================================
sns.set_theme(style="whitegrid", font="sans-serif")
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle(f"Phase 5 — Optimization & Purchase Intent Prediction ({best_model_name})", fontsize=16, fontweight="bold")

# 6a. Loss Convergence: BGD vs SGD vs Mini-Batch
ax = axes[0, 0]
epochs_arr = np.arange(1, n_epochs + 1)
ax.plot(epochs_arr, model_bgd.loss_history, "b-", linewidth=2, label="Batch GD (alpha=0.20)")
ax.plot(epochs_arr, model_sgd.loss_history, "r--", linewidth=1.5, label="SGD (alpha=0.01)")
ax.plot(epochs_arr, model_mbgd.loss_history, "g-.", linewidth=2, label="Mini-Batch GD (B=256, alpha=0.10)")
ax.set_title("Optimization: Loss Convergence Comparison", fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("Binary Cross-Entropy Loss")
ax.legend(frameon=True)

# 6b. Precision-Recall Curves (PR-AUC)
ax = axes[0, 1]
baseline_pr = np.mean(y_test)
colors = {"LightGBM": "#27ae60", "XGBoost": "#2980b9", "Random Forest": "#8e44ad", "Logistic Regression": "#e67e22"}

for name, probs in test_predictions.items():
    prec, rec, _ = precision_recall_curve(y_test, probs)
    ap = average_precision_score(y_test, probs)
    ax.plot(rec, prec, label=f"{name} (PR-AUC = {ap:.3f})", color=colors.get(name, "black"), linewidth=2)

ax.axhline(y=baseline_pr, color="gray", linestyle=":", label=f"Random Baseline ({baseline_pr*100:.1f}%)")
ax.set_title("Precision-Recall Curves (Key Imbalance Metric)", fontweight="bold")
ax.set_xlabel("Recall (True Positive Rate)")
ax.set_ylabel("Precision (Positive Predictive Value)")
ax.legend(loc="upper right", frameon=True)

# 6c. ROC Curves
ax = axes[1, 0]
for name, probs in test_predictions.items():
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_val = roc_auc_score(y_test, probs)
    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", color=colors.get(name, "black"), linewidth=2)

ax.plot([0, 1], [0, 1], "k--", label="Random Chance (0.50)")
ax.set_title("Receiver Operating Characteristic (ROC)", fontweight="bold")
ax.set_xlabel("False Positive Rate (1 - Specificity)")
ax.set_ylabel("True Positive Rate (Sensitivity)")
ax.legend(loc="lower right", frameon=True)

# 6d. Expected Profit Curve & Optimal Threshold
ax = axes[1, 1]
ax.plot(profit_df["threshold"], profit_df["expected_net_profit"], color="#2c3e50", linewidth=2.5, label="Expected Net Profit ($)")
ax.axvline(x=opt_tau, color="#e74c3c", linestyle="--", linewidth=2, label=f"Optimal tau* = {opt_tau:.2f} (${opt_profit:,.0f})")
ax.axvline(x=0.50, color="#95a5a6", linestyle=":", linewidth=1.5, label="Default tau = 0.50")
ax.scatter([opt_tau], [opt_profit], color="#e74c3c", s=80, zorder=5)
ax.set_title("Cost-Sensitive Profit Curve vs Decision Threshold", fontweight="bold")
ax.set_xlabel("Classification Decision Threshold (tau)")
ax.set_ylabel("Expected Marketing Net Profit ($)")
ax.legend(loc="upper right", frameon=True)

plt.tight_layout()
plot_path = REPORTS_DIR / "phase5_intent_prediction.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n📊 Phase 5 visualization saved: {plot_path}")

# ============================================================
# CELL 7 – Save Champion Model for Production Deployment
# ============================================================
champion_path = MODELS_DIR / "intent_model_champion.joblib"
joblib.dump(best_model, champion_path)
print(f"✅ Champion model saved to: {champion_path}")

print("\n" + "=" * 60)
print("PHASE 5 COMPLETE – SUMMARY")
print("=" * 60)
print(f"""
Supervised Model Benchmark (Holdout Test N = 22,192):
{models_benchmark_df.to_string(index=False)}

Commercial Threshold Optimization:
- Optimal Threshold (tau*)    : {opt_tau:.2f}
- Net Marketing Profit        : ${opt_profit:,.2f}
- Precision @ tau*            : {opt_precision*100:.1f}%
- Recall @ tau*               : {opt_recall*100:.1f}%

Next: Phase 6 will evaluate segment-level conversion differences (Q3)
and Lagrange Multiplier marketing budget optimization (Q5).
""")
