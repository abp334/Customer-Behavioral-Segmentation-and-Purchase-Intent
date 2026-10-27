# Tableau & Power BI Dashboard Integration Guide

This directory contains curated, clean, and normalized CSV datasets exported directly from the **DuckDB Gold Star Schema** (`data/gold/warehouse.duckdb`). They are pre-structured for immediate, drag-and-drop connection in **Tableau Desktop / Tableau Public** or **Microsoft Power BI**.

---

## 1. Exported Datasets Overview

| File | Entity Granularity | Row Count | Primary Use Case in Tableau / Power BI |
|:---|:---|:---:|:---|
| [`tableau_daily_metrics.csv`](file:///Users/aayushpandya/Desktop/Java/%20VS%20Code%20Projects/DSC_GP/dashboard/tableau_daily_metrics.csv) | Calendar Date | 31 | Executive traffic & revenue time series trends, day-of-week seasonality, and Average Order Value (AOV). |
| [`tableau_category_funnel.csv`](file:///Users/aayushpandya/Desktop/Java/%20VS%20Code%20Projects/DSC_GP/dashboard/tableau_category_funnel.csv) | Product Category | 11 | Behavioral funnel analysis (Views $\to$ Carts $\to$ Purchases), conversion rates by retail department, and category GMV contribution. |
| [`tableau_customer_segments.csv`](file:///Users/aayushpandya/Desktop/Java/%20VS%20Code%20Projects/DSC_GP/dashboard/tableau_customer_segments.csv) | Customer Profile | 10,000 | Customer persona profiling (Champions, Active Explorers, Window Shoppers, Dormant Bouncers), RFM scatter plots, and future purchase propensity. |
| [`tableau_lagrange_scenarios.csv`](file:///Users/aayushpandya/Desktop/Java/%20VS%20Code%20Projects/DSC_GP/dashboard/tableau_lagrange_scenarios.csv) | Budget Scenario | 5 | Marketing budget allocation scenarios ($5k to $100k), marginal returns ($\lambda^*$), net profit, and ROI comparisons. |

---

## 2. Direct DuckDB Live Connection (Alternative)

If you prefer a **Live Database Connection** rather than static CSV extracts:
1. In Tableau, select **Connect to Data** $\to$ **ODBC** or **Other Databases (JDBC)**.
2. Select the DuckDB driver and point the database file path to:
   `/Users/aayushpandya/Desktop/Java/ VS Code Projects/DSC_GP/data/gold/warehouse.duckdb`
3. You will have live SQL access to `dim_date`, `dim_product`, `dim_user`, `fact_events`, and `gold_user_features_segmented`.

---

## 3. Recommended Tableau Worksheets & Dashboard Layout

### Dashboard 1: Executive eCommerce & Funnel Overview
- **Sheet 1 (Top Banner KPIs)**:
  - Big Number Cards: Total Events (2.1M sampled / 42.45M raw), Active Customers (110,956), Total GMV ($10.8M), Baseline Conversion Rate (3.87%).
- **Sheet 2 (Dual-Axis Time Series)**:
  - X-axis: `date`. Left Y-axis: `total_events` (Line). Right Y-axis: `gmv` (Bar). Shows weekend surges.
- **Sheet 3 (Conversion Funnel Bar Chart)**:
  - Use `tableau_category_funnel.csv`.
  - Staged bars: Views $\to$ Carts $\to$ Purchases for Electronics, Appliances, Computers, etc.

### Dashboard 2: Customer Behavioral Personas (Q1 & Q3)
- **Sheet 1 (Persona Breakdown Donut Chart)**:
  - Slice by `persona_name`: Champions (10.2%), Active Explorers (21.2%), Recent Window Shoppers (32.7%), Dormant Bouncers (36.0%).
- **Sheet 2 (RFM Scatter Plot)**:
  - X-axis: `recency_days`, Y-axis: `total_spend` (log scale), Color: `persona_name`, Size: `total_sessions`.
- **Sheet 3 (Conversion Rate by Persona)**:
  - Bar chart showing 10-day conversion: Champions (16.9%) vs Active Explorers (4.9%) vs Window Shoppers (2.0%) vs Dormant (1.3%).

### Dashboard 3: Marketing Budget Optimization (Q5)
- **Sheet 1 (Stacked Budget Allocation Bar Chart)**:
  - Use `tableau_lagrange_scenarios.csv`.
  - X-axis: `campaign_budget` ($5k, $10k, $20k, $50k, $100k).
  - Stacked colors: Allocations to Champions, Active Explorers, Window Shoppers, and Dormant Bouncers.
- **Sheet 2 (ROI & Profitability Curve)**:
  - X-axis: `campaign_budget`, Y-axis: `expected_net_profit` and `expected_roi_pct`.

---

## 4. Calculated Fields for Tableau

Copy-paste these formulas into Tableau's Calculated Field editor if needed:

1. **Weekend vs Weekday Flag**:
   ```sql
   IF [is_weekend] = 1 THEN "Weekend" ELSE "Weekday" END
   ```

2. **Average Order Value (AOV)**:
   ```sql
   SUM([gmv]) / SUM([purchases])
   ```

3. **High Intent Customer Flag**:
   ```sql
   IF [will_purchase_next_10d] = 1 THEN "Future Buyer" ELSE "Non-Buyer" END
   ```
