"""
src/timeseries.py
=================
Time series extraction, stationarity diagnostics (ADF & KPSS),
autocorrelation analysis (ACF & PACF), and multi-model forecasting
(Holt-Winters, ARIMA, Seasonal SARIMA).
Answers Business Question Q4: "What will future traffic look like?"
"""

import numpy as np
import pandas as pd
import duckdb
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


def extract_daily_metrics(duckdb_path: str) -> pd.DataFrame:
    """
    Extract aggregated daily clickstream metrics from DuckDB Star Schema:
      - full_date: Calendar date (YYYY-MM-DD)
      - total_events: Sum of views, carts, and purchases
      - active_users: Count of distinct customers active on that day
      - total_purchases: Number of completed purchase transactions
      - total_gmv: Gross merchandise value (USD) generated from purchases
      - is_weekend: Binary indicator for Saturday/Sunday
    """
    con = duckdb.connect(str(duckdb_path), read_only=True)
    query = """
    SELECT 
        d.full_date,
        d.day_name,
        d.is_weekend,
        count(*) AS total_events,
        count(distinct f.user_id) AS active_users,
        count(*) FILTER (WHERE f.event_type = 'purchase') AS total_purchases,
        round(coalesce(sum(f.price) FILTER (WHERE f.event_type = 'purchase'), 0.0), 2) AS total_gmv
    FROM fact_events f
    JOIN dim_date d ON f.date_key = d.date_key
    GROUP BY d.full_date, d.day_name, d.is_weekend
    ORDER BY d.full_date;
    """
    df_daily = con.execute(query).df()
    df_daily["full_date"] = pd.to_datetime(df_daily["full_date"])
    con.close()
    return df_daily


def run_stationarity_suite(series: pd.Series, series_name: str) -> pd.DataFrame:
    """
    Execute dual stationarity diagnostics:
      1. Augmented Dickey-Fuller (ADF) Test
         H0: Series has a unit root (Non-stationary)
         H1: Series is stationary
      2. KPSS Test (Kwiatkowski-Phillips-Schmidt-Shin)
         H0: Series is stationary
         H1: Series has a unit root (Non-stationary)
    Evaluates both raw series and first-differenced series.
    """
    s_raw = series.dropna()
    s_diff = series.diff().dropna()

    records = []
    for s, name in [(s_raw, f"{series_name} (Raw)"), (s_diff, f"{series_name} (1st Diff)")]:
        # ADF Test
        adf_out = adfuller(s, autolag="AIC")
        adf_stat, adf_pval = adf_out[0], adf_out[1]
        
        # KPSS Test
        kpss_out = kpss(s, regression="c", nlags="auto")
        kpss_stat, kpss_pval = kpss_out[0], kpss_out[1]

        records.append({
            "series": name,
            "adf_statistic": round(float(adf_stat), 4),
            "adf_pvalue": round(float(adf_pval), 4),
            "adf_decision": "Stationary (Reject H0)" if adf_pval < 0.05 else "Non-Stationary (Fail to Reject)",
            "kpss_statistic": round(float(kpss_stat), 4),
            "kpss_pvalue": round(float(kpss_pval), 4),
            "kpss_decision": "Stationary (Fail to Reject)" if kpss_pval >= 0.05 else "Non-Stationary (Reject H0)"
        })

    return pd.DataFrame(records)


def evaluate_forecast_models(series: pd.Series,
                            train_len: int = 24,
                            test_len: int = 7) -> tuple[pd.DataFrame, dict]:
    """
    Fit and evaluate 4 forecasting models over out-of-sample holdout test window:
      1. Naive Seasonal Baseline (y_{t-7})
      2. Holt-Winters Triple Exponential Smoothing (Additive Trend & 7-day Seasonality)
      3. ARIMA(1, 1, 1) Non-Seasonal
      4. Seasonal SARIMA(1, 1, 1) x (1, 0, 0)_7
    Returns: evaluation benchmark DataFrame, dictionary of model predictions
    """
    train = series.iloc[:train_len].values
    test = series.iloc[train_len:train_len + test_len].values

    predictions = {}

    # 1. Naive 7-day Seasonal Baseline
    pred_naive = series.iloc[train_len - 7:train_len].values
    predictions["Seasonal Baseline"] = pred_naive

    # 2. Holt-Winters Exponential Smoothing
    hw_model = ExponentialSmoothing(
        train, trend="add", seasonal="add", seasonal_periods=7
    ).fit()
    pred_hw = hw_model.forecast(steps=test_len)
    predictions["Holt-Winters"] = pred_hw

    # 3. ARIMA(1, 1, 1) Non-Seasonal
    arima_model = SARIMAX(
        train, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False
    ).fit(disp=False)
    pred_arima = arima_model.forecast(steps=test_len)
    predictions["ARIMA(1,1,1)"] = pred_arima

    # 4. Seasonal SARIMA(1, 1, 1) x (1, 0, 0)_7
    sarima_model = SARIMAX(
        train, order=(1, 1, 1), seasonal_order=(1, 0, 0, 7),
        enforce_stationarity=False, enforce_invertibility=False
    ).fit(disp=False)
    pred_sarima = sarima_model.forecast(steps=test_len)
    predictions["SARIMA(1,1,1)x(1,0,0)7"] = pred_sarima

    # Calculate Evaluation Metrics
    benchmarks = []
    for model_name, y_hat in predictions.items():
        rmse = float(np.sqrt(np.mean((test - y_hat) ** 2)))
        mae = float(np.mean(np.abs(test - y_hat)))
        mape = float(np.mean(np.abs((test - y_hat) / test)) * 100.0)

        benchmarks.append({
            "model": model_name,
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "mape_pct": round(mape, 2)
        })

    benchmarks_df = pd.DataFrame(benchmarks).sort_values("mape_pct")
    return benchmarks_df, predictions
