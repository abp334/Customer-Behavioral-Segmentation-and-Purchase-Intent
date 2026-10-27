"""
src/budget_optimization.py
==========================
Constrained optimization using Lagrange Multipliers for marketing budget allocation.
Directly aligns with Syllabus Unit 7 (Optimization & Lagrange Multipliers).
Answers Business Question Q5: "What should the business do?"
"""

import numpy as np
import pandas as pd
import scipy.optimize as opt


def solve_lagrange_budget(alphas: np.ndarray,
                         betas: np.ndarray,
                         total_budget: float = 10000.0) -> tuple[np.ndarray, float]:
    """
    Solve constrained optimization problem:
      max sum_{i=1}^k alpha_i * ln(1 + beta_i * b_i)
      subject to: sum b_i = total_budget, b_i >= 0

    Lagrangian:
      L(b, lambda) = sum alpha_i * ln(1 + beta_i * b_i) - lambda * (sum b_i - B)

    KKT / Optimality Condition:
      dL/db_i = (alpha_i * beta_i) / (1 + beta_i * b_i) - lambda = 0
      => b_i*(lambda) = max(0, alpha_i / lambda - 1 / beta_i)

    Returns:
      b_star: optimal budget vector (allocations)
      lambda_star: optimal shadow price (marginal return per dollar)
    """
    alphas = np.asarray(alphas, dtype=float)
    betas = np.asarray(betas, dtype=float)

    def budget_constraint(lam):
        b = np.maximum(0.0, alphas / lam - 1.0 / betas)
        return float(np.sum(b) - total_budget)

    # Solve for lambda using Brent's method
    # Lower bound lambda_min > 0, Upper bound lambda_max sufficiently large
    sol = opt.root_scalar(budget_constraint, bracket=[1e-4, 1e5], method="brentq")
    lambda_star = float(sol.root)
    b_star = np.maximum(0.0, alphas / lambda_star - 1.0 / betas)

    return b_star, lambda_star


def revenue_function(b: np.ndarray, alphas: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """Concave diminishing-returns revenue: R_i(b_i) = alpha_i * ln(1 + beta_i * b_i)."""
    return alphas * np.log1p(betas * b)


def compare_budget_strategies(segment_names: list[str],
                              customer_counts: list[int],
                              alphas: np.ndarray,
                              betas: np.ndarray,
                              total_budget: float = 10000.0) -> pd.DataFrame:
    """
    Compare 3 budget allocation strategies:
      1. Lagrange Optimal Allocation
      2. Naive Uniform Allocation (Equal split)
      3. Size-Proportional Allocation (Allocated by customer population)
    """
    k = len(segment_names)
    b_lagrange, lam_star = solve_lagrange_budget(alphas, betas, total_budget)
    
    # Strategy 2: Uniform
    b_uniform = np.full(k, total_budget / k)
    
    # Strategy 3: Size-proportional
    counts = np.asarray(customer_counts, dtype=float)
    b_size = total_budget * (counts / np.sum(counts))
    
    rev_lagrange = revenue_function(b_lagrange, alphas, betas)
    rev_uniform = revenue_function(b_uniform, alphas, betas)
    rev_size = revenue_function(b_size, alphas, betas)
    
    records = []
    strategies = [
        ("Lagrange Multiplier Optimal", b_lagrange, rev_lagrange),
        ("Uniform Equal Split", b_uniform, rev_uniform),
        ("Size-Proportional Allocation", b_size, rev_size)
    ]
    
    for strat_name, b_vec, r_vec in strategies:
        tot_rev = float(np.sum(r_vec))
        net_profit = tot_rev - total_budget
        roi = (net_profit / total_budget) * 100.0
        
        row_dict = {
            "strategy": strat_name,
            "total_budget": total_budget,
            "total_revenue": round(tot_rev, 2),
            "net_profit": round(net_profit, 2),
            "roi_pct": round(roi, 2)
        }
        for i, sname in enumerate(segment_names):
            short_sname = sname.split("(")[0].strip()
            row_dict[f"budget_{short_sname}"] = round(float(b_vec[i]), 2)
            row_dict[f"rev_{short_sname}"] = round(float(r_vec[i]), 2)
            
        records.append(row_dict)
        
    return pd.DataFrame(records), b_lagrange, lam_star
