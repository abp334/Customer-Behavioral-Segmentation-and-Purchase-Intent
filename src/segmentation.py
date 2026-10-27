"""
src/segmentation.py
===================
Linear Algebra decompositions, dimensionality reduction,
and unsupervised customer behavioral clustering.
Covers Syllabus Units 4 & 5 (Linear Algebra) and Unit 10 (Unsupervised Learning).
"""

import numpy as np
import pandas as pd
import scipy.linalg as la
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score


def prepare_feature_matrix(df: pd.DataFrame,
                           feature_cols: list[str],
                           log_cols: list[str]) -> tuple[np.ndarray, StandardScaler, list[str]]:
    """
    Prepare centered and standardized design matrix X:
      1. Apply log1p transformation to skewed count and monetary features
      2. Impute any remaining NaNs with 0
      3. Standardize to zero mean and unit variance: z = (x - mu) / sigma
    """
    X_df = df[feature_cols].copy()
    for col in log_cols:
        if col in X_df.columns:
            X_df[col] = np.log1p(np.maximum(0, X_df[col].fillna(0)))
            
    X_df = X_df.fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values)
    return X_scaled, scaler, feature_cols


def run_linear_algebra_verifications(X_scaled: np.ndarray) -> dict:
    """
    Execute and mathematically verify key Linear Algebra decompositions:
      1. Covariance Matrix: C = (1/(n-1)) * X_c^T * X_c
      2. SVD: X = U * Sigma * V^T
      3. Eigendecomposition of C: C * v = lambda * v
      4. Theoretical link: lambda_i == sigma_i^2 / (n - 1)
      5. QR Decomposition: X_sub = Q * R, verifying Q^T * Q == I
      6. Cholesky Factorization: C_reg = L * L^T, verifying positive definiteness
    """
    n, p = X_scaled.shape
    
    # 1. Sample Covariance Matrix (X_scaled is already centered: mean = 0)
    C = (X_scaled.T @ X_scaled) / (n - 1)
    
    # Symmetry check: ||C - C^T||_F
    sym_error = float(np.linalg.norm(C - C.T, ord="fro"))
    
    # 2. Singular Value Decomposition of X_scaled
    U, S, Vt = la.svd(X_scaled, full_matrices=False)
    V = Vt.T
    
    # 3. Eigendecomposition of C
    eigvals, eigvecs = la.eigh(C)
    # Sort eigenvalues descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # 4. Mathematical Identity: lambda_i == sigma_i^2 / (n - 1)
    svd_lambdas = (S ** 2) / (n - 1)
    max_eigen_diff = float(np.max(np.abs(eigvals - svd_lambdas)))
    
    # 5. QR Decomposition on a representative subset (for numerical efficiency)
    sample_size = min(5000, n)
    X_sub = X_scaled[:sample_size, :]
    Q, R = la.qr(X_sub, mode="economic")
    # Orthogonality check: ||Q^T * Q - I||_F
    I_p = np.eye(p)
    qr_ortho_error = float(np.linalg.norm(Q.T @ Q - I_p, ord="fro"))
    
    # 6. Cholesky Factorization of Regularized Covariance Matrix
    # (Add tiny epsilon * I to ensure strict positive definiteness)
    C_reg = C + 1e-6 * np.eye(p)
    L = la.cholesky(C_reg, lower=True)
    chol_reconstruction_error = float(np.linalg.norm(C_reg - L @ L.T, ord="fro"))
    
    return {
        "n_samples": n,
        "n_features": p,
        "covariance_symmetry_error": round(sym_error, 12),
        "svd_eigenvalue_max_diff": round(max_eigen_diff, 12),
        "qr_orthogonality_error": round(qr_ortho_error, 12),
        "cholesky_reconstruction_error": round(chol_reconstruction_error, 12),
        "singular_values": [round(float(s), 2) for s in S[:6]],
        "eigenvalues": [round(float(e), 4) for e in eigvals[:6]],
        "explained_variance_ratio": [round(float(ev / p * 100), 2) for ev in eigvals[:6]],
        "cumulative_variance_ratio": [round(float(c), 2) for c in np.cumsum(eigvals / p * 100)[:6]]
    }


def compute_pca_projection(X_scaled: np.ndarray,
                           feature_names: list[str],
                           n_components: int = 5) -> tuple[PCA, np.ndarray, pd.DataFrame]:
    """
    Project centered data onto top Principal Components.
    Returns: fitted PCA model, projected matrix Z, loadings DataFrame
    """
    pca = PCA(n_components=n_components, random_state=42)
    Z = pca.fit_transform(X_scaled)
    
    pc_cols = [f"PC{i+1}" for i in range(n_components)]
    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=pc_cols
    )
    return pca, Z, loadings_df


def evaluate_cluster_k(Z: np.ndarray,
                       k_range=range(2, 8),
                       sample_size: int = 10000,
                       random_state: int = 42) -> pd.DataFrame:
    """
    Evaluate K-Means across k = 2..7 using:
      1. Inertia (Within-Cluster Sum of Squares)
      2. Silhouette Score (Cluster cohesion vs separation)
      3. Davies-Bouldin Index (Lower is better)
    """
    rng = np.random.default_rng(random_state)
    sample_idx = rng.choice(len(Z), size=min(sample_size, len(Z)), replace=False)
    Z_sample = Z[sample_idx]
    
    records = []
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
        km.fit(Z)
        
        inertia = float(km.inertia_)
        labels_sample = km.predict(Z_sample)
        sil = float(silhouette_score(Z_sample, labels_sample))
        db = float(davies_bouldin_score(Z_sample, labels_sample))
        
        records.append({
            "k": k,
            "inertia": round(inertia, 2),
            "silhouette_score": round(sil, 4),
            "davies_bouldin_index": round(db, 4)
        })
    return pd.DataFrame(records)


def fit_final_segmentation(Z: np.ndarray, k: int = 4, random_state: int = 42) -> tuple[KMeans, np.ndarray]:
    """Fit final K-Means model on projected features and return model and labels."""
    km = KMeans(n_clusters=k, init="k-means++", n_init=15, random_state=random_state)
    labels = km.fit_predict(Z)
    return km, labels


def profile_customer_segments(df: pd.DataFrame, cluster_labels: np.ndarray) -> pd.DataFrame:
    """
    Aggregate customer features by cluster and assign human-interpretable persona labels.
    """
    df_profile = df.copy()
    df_profile["cluster"] = cluster_labels
    
    # Compute cluster statistics
    agg_dict = {
        "user_id": "count",
        "recency_days": "median",
        "tenure_days": "median",
        "total_sessions": "median",
        "total_events": "median",
        "view_count": "median",
        "cart_count": "median",
        "purchase_count": "mean",
        "total_spend": "mean",
        "avg_purchase_price": "mean",
        "n_unique_products": "median",
        "will_purchase_next_10d": "mean"
    }
    
    summary = df_profile.groupby("cluster").agg(agg_dict).reset_index()
    summary.rename(columns={
        "user_id": "customer_count",
        "will_purchase_next_10d": "next_10d_conversion_rate"
    }, inplace=True)
    summary["pct_of_total"] = (summary["customer_count"] / len(df) * 100).round(2)
    summary["next_10d_conversion_rate"] = (summary["next_10d_conversion_rate"] * 100).round(2)
    summary["total_spend"] = summary["total_spend"].round(2)
    summary["avg_purchase_price"] = summary["avg_purchase_price"].round(2)
    summary["purchase_count"] = summary["purchase_count"].round(3)
    
    # Dynamically assign intuitive business persona names based on spend and activity
    def assign_persona(row):
        spend = row["total_spend"]
        conv = row["next_10d_conversion_rate"]
        rec = row["recency_days"]
        events = row["total_events"]
        
        if spend > 150 or conv > 10.0:
            return "Champions (High-Value Spenders)"
        elif events >= 15 or conv >= 4.0:
            return "Active Explorers (High Engagement)"
        elif rec >= 8:
            return "Dormant Bouncers (Cold / Inactive)"
        else:
            return "Recent Window Shoppers (Warm Bouncers)"
            
    summary["persona_name"] = summary.apply(assign_persona, axis=1)
    
    return summary
