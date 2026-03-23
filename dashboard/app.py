"""Earnings Alpha dashboard with robust demo-data fallback."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Earnings Alpha", layout="wide")
st.title("Alpha Signal Extraction from Earnings Calls")
st.caption("NLP signals, causal estimates, forecasting metrics, and portfolio backtest outcomes.")


FEATURES_DIR = Path("data/processed/features")
RETURNS_DIR = Path("data/processed/returns")


def _demo_sentiment_per_call() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 180
    tickers = np.array(["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"])
    event_dates = pd.date_range("2023-01-01", periods=n, freq="7D")
    uncertainty = np.clip(rng.normal(0.14, 0.035, n), 0.03, 0.35)
    sentiment = np.clip(rng.normal(0.08, 0.09, n), -0.25, 0.35)
    novelty = np.clip(rng.normal(0.42, 0.15, n), 0.05, 0.95)
    car_20d = 0.03 * sentiment - 0.05 * uncertainty + rng.normal(0.0, 0.03, n)
    return pd.DataFrame(
        {
            "call_id": [f"call_{i:04d}" for i in range(n)],
            "ticker": rng.choice(tickers, size=n),
            "event_date": event_dates,
            "sentiment_score": sentiment,
            "uncertainty_ratio": uncertainty,
            "topic_novelty": novelty,
            "car_20d": car_20d,
        }
    )


def _demo_did_estimates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"variable": "uncertainty_ratio", "coef": -0.0182, "std_err": 0.0045, "t_stat": -4.04, "p_value": 0.0001},
            {"variable": "sentiment_score", "coef": 0.0127, "std_err": 0.0038, "t_stat": 3.31, "p_value": 0.0009},
            {"variable": "topic_novelty", "coef": -0.0061, "std_err": 0.0027, "t_stat": -2.26, "p_value": 0.0240},
        ]
    )


def _demo_car_event_study() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for ticker in ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"]:
        for window in ["[0,1]", "[0,5]", "[0,20]"]:
            base = {"[0,1]": 0.002, "[0,5]": 0.005, "[0,20]": 0.011}[window]
            car = base + float(rng.normal(0.0, 0.01))
            rows.append(
                {
                    "ticker": ticker,
                    "event_window": window,
                    "car": car,
                    "ar_mean": car / max(1, int(window.split(",")[1].strip("]")) + 1),
                    "n_days": int(window.split(",")[1].strip("]")) + 1,
                    "t_stat": car / 0.01,
                }
            )
    return pd.DataFrame(rows)


def _demo_model_comparison() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"model": "Random Walk", "RMSE": 0.0931, "IC": 0.021, "hit_rate": 0.511},
            {"model": "SUE OLS", "RMSE": 0.0874, "IC": 0.063, "hit_rate": 0.531},
            {"model": "XGBoost", "RMSE": 0.0748, "IC": 0.119, "hit_rate": 0.572},
            {"model": "TFT", "RMSE": 0.0694, "IC": 0.147, "hit_rate": 0.589},
        ]
    )


def _demo_backtest_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "annualised_sharpe": 1.34,
                "annualised_return_ls": 0.128,
                "max_drawdown": -0.084,
                "turnover": 0.61,
                "transaction_cost_bps": 5.0,
                "n_long": 82,
                "n_short": 82,
            }
        ]
    )


def _demo_equity_curve() -> pd.DataFrame:
    rng = np.random.default_rng(99)
    dates = pd.date_range("2023-01-01", periods=120, freq="W")
    returns = rng.normal(0.0019, 0.012, len(dates))
    eq = (1 + pd.Series(returns, index=dates)).cumprod()
    return pd.DataFrame({"date": dates, "equity_curve": eq.values})


def load_or_demo(path: Path, demo_func, use_demo: bool) -> tuple[pd.DataFrame, str]:
    if path.exists():
        return pd.read_parquet(path), "pipeline"
    if use_demo:
        return demo_func(), "demo"
    return pd.DataFrame(), "missing"


st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Signal Explorer", "Causal Estimates", "Model Performance", "Backtest Results", "Methodology"],
)
use_demo = st.sidebar.toggle("Use demo data if files are missing", value=True)

sentiment_df, sentiment_src = load_or_demo(
    FEATURES_DIR / "sentiment_per_call.parquet", _demo_sentiment_per_call, use_demo
)
did_df, did_src = load_or_demo(RETURNS_DIR / "did_estimates.parquet", _demo_did_estimates, use_demo)
car_df, car_src = load_or_demo(RETURNS_DIR / "car_event_study.parquet", _demo_car_event_study, use_demo)
model_df, model_src = load_or_demo(
    FEATURES_DIR / "model_comparison.parquet", _demo_model_comparison, use_demo
)
backtest_df, bt_src = load_or_demo(
    RETURNS_DIR / "backtest_results.parquet", _demo_backtest_results, use_demo
)
equity_df = _demo_equity_curve()

source_badges = {
    "Signal data": sentiment_src,
    "Causal data": did_src,
    "Model data": model_src,
    "Backtest data": bt_src,
}

if page == "Overview":
    st.subheader("System Overview")
    st.write(
        "This dashboard summarizes transcript-derived signals, causal estimates, model quality, "
        "and long/short strategy performance."
    )
    cols = st.columns(4)
    total_calls = int(len(sentiment_df)) if not sentiment_df.empty else 0
    avg_sentiment = float(sentiment_df["sentiment_score"].mean()) if "sentiment_score" in sentiment_df else 0.0
    avg_uncertainty = float(sentiment_df["uncertainty_ratio"].mean()) if "uncertainty_ratio" in sentiment_df else 0.0
    sharpe = float(backtest_df["annualised_sharpe"].iloc[0]) if "annualised_sharpe" in backtest_df else 0.0
    cols[0].metric("Calls analyzed", f"{total_calls:,}")
    cols[1].metric("Avg sentiment", f"{avg_sentiment:.3f}")
    cols[2].metric("Avg uncertainty", f"{avg_uncertainty:.3f}")
    cols[3].metric("Strategy Sharpe", f"{sharpe:.2f}")
    st.caption("Data source status: " + " | ".join([f"{k}: {v}" for k, v in source_badges.items()]))

    c1, c2 = st.columns(2)
    with c1:
        if not sentiment_df.empty and {"event_date", "sentiment_score", "uncertainty_ratio"}.issubset(sentiment_df.columns):
            tmp = sentiment_df.copy()
            tmp["event_date"] = pd.to_datetime(tmp["event_date"])
            monthly = (
                tmp.set_index("event_date")[["sentiment_score", "uncertainty_ratio"]]
                .resample("M")
                .mean()
                .reset_index()
            )
            st.markdown("**Monthly signal trend**")
            st.line_chart(monthly.set_index("event_date"))
    with c2:
        if not equity_df.empty:
            st.markdown("**Strategy equity curve (index = 1.0)**")
            st.line_chart(equity_df.set_index("date"))

elif page == "Signal Explorer":
    st.subheader("Signal Explorer")
    if sentiment_df.empty:
        st.warning("No signal data available. Enable demo mode or run NLP pipelines.")
    else:
        tickers = sorted(sentiment_df["ticker"].dropna().unique().tolist()) if "ticker" in sentiment_df else []
        selected = st.multiselect("Tickers", tickers, default=tickers[:4] if tickers else [])
        filtered = sentiment_df[sentiment_df["ticker"].isin(selected)] if selected else sentiment_df
        st.dataframe(filtered.head(200), use_container_width=True)
        numeric = [c for c in ["sentiment_score", "uncertainty_ratio", "topic_novelty", "car_20d"] if c in filtered.columns]
        if numeric:
            st.markdown("**Signal distributions**")
            st.bar_chart(filtered[numeric].describe().T[["mean", "std"]])
        if {"sentiment_score", "car_20d"}.issubset(filtered.columns):
            st.markdown("**Sentiment vs 20D CAR**")
            st.scatter_chart(filtered, x="sentiment_score", y="car_20d")

elif page == "Causal Estimates":
    st.subheader("Causal Estimates")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Difference-in-Differences (sample)**")
        if did_df.empty:
            st.warning("No DiD estimates available.")
        else:
            st.dataframe(did_df, use_container_width=True)
            if {"variable", "coef"}.issubset(did_df.columns):
                st.bar_chart(did_df.set_index("variable")[["coef"]])
    with c2:
        st.markdown("**Event Study CAR by window**")
        if car_df.empty:
            st.warning("No event study results available.")
        else:
            st.dataframe(car_df.head(60), use_container_width=True)
            if {"event_window", "car"}.issubset(car_df.columns):
                st.bar_chart(car_df.groupby("event_window")["car"].mean())

elif page == "Model Performance":
    st.subheader("Model Performance")
    if model_df.empty:
        st.warning("No model comparison file found. Enable demo mode or run forecasting.")
    else:
        st.dataframe(model_df, use_container_width=True)
        if {"model", "RMSE", "IC", "hit_rate"}.issubset(model_df.columns):
            c1, c2, c3 = st.columns(3)
            c1.bar_chart(model_df.set_index("model")[["RMSE"]])
            c2.bar_chart(model_df.set_index("model")[["IC"]])
            c3.bar_chart(model_df.set_index("model")[["hit_rate"]])

elif page == "Backtest Results":
    st.subheader("Backtest Results")
    if backtest_df.empty:
        st.warning("No backtest results found. Enable demo mode or run portfolio simulation.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Annualized Sharpe", f"{float(backtest_df['annualised_sharpe'].iloc[0]):.2f}")
        c2.metric("Annualized L/S Return", f"{float(backtest_df['annualised_return_ls'].iloc[0]) * 100:.2f}%")
        if "max_drawdown" in backtest_df:
            c3.metric("Max Drawdown", f"{float(backtest_df['max_drawdown'].iloc[0]) * 100:.2f}%")
        if "turnover" in backtest_df:
            c4.metric("Turnover", f"{float(backtest_df['turnover'].iloc[0]):.2f}")
        st.dataframe(backtest_df, use_container_width=True)
        st.markdown("**Equity curve**")
        st.line_chart(equity_df.set_index("date"))

elif page == "Methodology":
    st.subheader("Methodology and Caveats")
    st.markdown(
        """
        - **Signal generation**: transcript-level sentiment, uncertainty, and topic novelty features.
        - **Causal analysis**: event study and DiD-style coefficient estimates.
        - **Forecasting**: model comparison across baselines and ML models (e.g., XGBoost/TFT).
        - **Portfolio simulation**: long top-quantile / short bottom-quantile signal strategy.
        """
    )
    st.info(
        "This deployment is an MVP research dashboard. Some pages may show demo data when "
        "pipeline artifacts are unavailable."
    )
