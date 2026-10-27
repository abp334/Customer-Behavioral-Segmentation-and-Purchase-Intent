"""
src/stats.py
============
Statistical inference, hypothesis testing, confidence intervals,
and distribution fitting utilities for eCommerce analytics.
Aligns directly with Syllabus Unit 3 (Statistics & Probability).
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint


def calc_summary_stats(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Calculate comprehensive descriptive statistics:
      - Central tendency: mean, median, 5% trimmed mean
      - Dispersion: std, IQR, MAD (median absolute deviation), CV (coeff of variation)
      - Shape: skewness, kurtosis
      - Extremes: min, max, Q1 (25th), Q3 (75th)
    """
    records = []
    for col in columns:
        s = df[col].dropna()
        n = len(s)
        if n == 0:
            continue
        
        mean_val = float(s.mean())
        std_val = float(s.std(ddof=1))
        median_val = float(s.median())
        q25, q75 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr_val = q75 - q25
        trimmed_mean = float(stats.trim_mean(s, proportiontocut=0.05))
        mad_val = float(stats.median_abs_deviation(s))
        skew_val = float(stats.skew(s))
        kurt_val = float(stats.kurtosis(s))
        cv_val = std_val / mean_val if mean_val != 0 else np.nan

        records.append({
            "feature": col,
            "count": n,
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "median": round(median_val, 4),
            "trimmed_mean_5pct": round(trimmed_mean, 4),
            "iqr": round(iqr_val, 4),
            "mad": round(mad_val, 4),
            "skewness": round(skew_val, 4),
            "kurtosis": round(kurt_val, 4),
            "cv": round(cv_val, 4),
            "min": round(float(s.min()), 4),
            "q25": round(q25, 4),
            "q75": round(q75, 4),
            "max": round(float(s.max()), 4),
        })

    return pd.DataFrame(records)


def bootstrap_ci(data: np.ndarray | pd.Series,
                 stat_func=np.mean,
                 n_boot: int = 2000,
                 ci: float = 0.95,
                 random_state: int = 42) -> tuple[float, float, float]:
    """
    Compute non-parametric bootstrap confidence interval.
    Returns: (point_estimate, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(random_state)
    arr = np.asarray(data)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    point_est = float(stat_func(arr))
    
    # Generate bootstrap samples
    boot_samples = rng.choice(arr, size=(n_boot, n), replace=True)
    boot_stats = np.apply_along_axis(stat_func, 1, boot_samples)
    
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_stats, 100 * alpha))
    upper = float(np.percentile(boot_stats, 100 * (1.0 - alpha)))
    
    return point_est, lower, upper


def wilson_proportion_ci(count: int, nobs: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """
    Compute Wilson score interval for a binomial proportion.
    Superior to normal approximation (Wald) near boundary proportions (p < 0.05).
    Returns: (point_estimate, lower_bound, upper_bound)
    """
    p = count / nobs if nobs > 0 else 0.0
    alpha = 1.0 - confidence
    lower, upper = proportion_confint(count, nobs, alpha=alpha, method="wilson")
    return float(p), float(lower), float(upper)


def welch_two_sample_ttest(sample_a: np.ndarray | pd.Series,
                           sample_b: np.ndarray | pd.Series) -> dict:
    """
    Perform Welch's two-sample t-test (robust against unequal sample sizes and variances).
    H0: mu_a == mu_b
    H1: mu_a != mu_b
    """
    a = np.asarray(sample_a)
    b = np.asarray(sample_b)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    
    stat, p_val = stats.ttest_ind(a, b, equal_var=False)
    
    # Cohen's d effect size with pooled std
    n1, n2 = len(a), len(b)
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    cohens_d = (np.mean(a) - np.mean(b)) / pooled_sd if pooled_sd != 0 else 0.0
    
    return {
        "test": "Welch's Two-Sample t-test",
        "t_statistic": round(float(stat), 4),
        "p_value": float(p_val),
        "mean_group_a": round(float(np.mean(a)), 4),
        "mean_group_b": round(float(np.mean(b)), 4),
        "cohens_d": round(float(cohens_d), 4),
        "reject_h0_alpha_05": p_val < 0.05
    }


def mann_whitney_u_test(sample_a: np.ndarray | pd.Series,
                        sample_b: np.ndarray | pd.Series) -> dict:
    """
    Perform Mann-Whitney U non-parametric rank test for stochastic dominance / medians.
    Ideal for heavily skewed eCommerce metrics (spend, views, sessions).
    """
    a = np.asarray(sample_a)
    b = np.asarray(sample_b)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    
    stat, p_val = stats.mannwhitneyu(a, b, alternative="two-sided")
    
    # Rank-biserial correlation effect size r = 1 - 2*U / (n1*n2)
    n1, n2 = len(a), len(b)
    rank_biserial = 1.0 - (2.0 * stat / (n1 * n2))
    
    return {
        "test": "Mann-Whitney U Test",
        "u_statistic": float(stat),
        "p_value": float(p_val),
        "median_group_a": round(float(np.median(a)), 4),
        "median_group_b": round(float(np.median(b)), 4),
        "rank_biserial_r": round(float(rank_biserial), 4),
        "reject_h0_alpha_05": p_val < 0.05
    }


def chi_square_independence_test(df: pd.DataFrame, col_x: str, col_y: str) -> dict:
    """
    Perform Pearson's Chi-square test of independence on two categorical variables.
    Computes contingency table, Chi2 stat, p-value, degrees of freedom, and Cramér's V.
    """
    ct = pd.crosstab(df[col_x], df[col_y])
    chi2, p_val, dof, expected = stats.chi2_contingency(ct)
    
    # Cramér's V effect size
    n = ct.sum().sum()
    min_dim = min(ct.shape[0] - 1, ct.shape[1] - 1)
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 and n > 0 else 0.0
    
    return {
        "test": "Chi-Square Test of Independence",
        "chi2_statistic": round(float(chi2), 4),
        "p_value": float(p_val),
        "degrees_of_freedom": int(dof),
        "cramers_v": round(float(cramers_v), 4),
        "reject_h0_alpha_05": p_val < 0.05,
        "contingency_table": ct
    }


def evaluate_distribution_fit(data: np.ndarray | pd.Series,
                             dist_names=("norm", "lognorm", "expon", "pareto")) -> pd.DataFrame:
    """
    Fit theoretical probability distributions to continuous empirical data.
    Evaluates goodness of fit via Kolmogorov-Smirnov (KS) test statistic and Log-Likelihood.
    """
    arr = np.asarray(data)
    arr = arr[~np.isnan(arr)]
    # Filter strictly positive values for Log-Normal and Pareto
    arr_pos = arr[arr > 0]
    n = len(arr_pos)
    
    results = []
    for dname in dist_names:
        dist_obj = getattr(stats, dname)
        try:
            params = dist_obj.fit(arr_pos)
            ks_stat, ks_pval = stats.kstest(arr_pos, dname, args=params)
            loglik = float(np.sum(dist_obj.logpdf(arr_pos, *params)))
            k = len(params)
            aic = 2 * k - 2 * loglik
            bic = k * np.log(n) - 2 * loglik
            
            results.append({
                "distribution": dname,
                "ks_statistic": round(float(ks_stat), 4),
                "ks_pvalue": float(ks_pval),
                "log_likelihood": round(loglik, 2),
                "aic": round(aic, 2),
                "bic": round(bic, 2),
                "parameters": str([round(float(p), 4) for p in params])
            })
        except Exception as e:
            continue
            
    df_res = pd.DataFrame(results).sort_values("ks_statistic")
    return df_res
