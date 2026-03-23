"""
Main Streamlit app: signal explorer, causal estimates, model performance, backtest results.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Earnings Alpha", layout="wide")
st.title("Alpha Signal Extraction from Earnings Calls")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Signal Explorer", "Causal Estimates", "Model Performance", "Backtest Results"],
)

if page == "Overview":
    st.markdown("""
    This dashboard explores NLP-derived alpha signals from earnings call transcripts.
    - **Signal Explorer**: Per-call NLP signal breakdown (sentiment, topics, uncertainty).
    - **Causal Estimates**: DiD / event study results.
    - **Model Performance**: Forecast accuracy and SHAP importance.
    - **Backtest Results**: L/S portfolio performance.
    """)
    config_path = Path("configs/data_config.yaml")
    if config_path.exists():
        st.success("Config found. Run ingestion and NLP pipelines to populate data.")

elif page == "Signal Explorer":
    st.header("Signal Explorer")
    features_dir = Path("data/processed/features")
    if (features_dir / "sentiment_per_call.parquet").exists():
        import pandas as pd
        df = pd.read_parquet(features_dir / "sentiment_per_call.parquet")
        st.dataframe(df.head(100))
        st.line_chart(df.select_dtypes(include="number").iloc[:50])
    else:
        st.info("Run NLP pipelines to generate sentiment_per_call.parquet.")

elif page == "Causal Estimates":
    st.header("Causal Estimates")
    returns_dir = Path("data/processed/returns")
    if (returns_dir / "did_estimates.parquet").exists():
        import pandas as pd
        df = pd.read_parquet(returns_dir / "did_estimates.parquet")
        st.dataframe(df)
    else:
        st.info("Run causal/diff_in_diff to generate did_estimates.parquet.")

elif page == "Model Performance":
    st.header("Model Performance")
    features_dir = Path("data/processed/features")
    if (features_dir / "model_comparison.parquet").exists():
        import pandas as pd
        df = pd.read_parquet(features_dir / "model_comparison.parquet")
        st.dataframe(df)
        st.bar_chart(df.set_index("model")[["RMSE", "IC", "hit_rate"]])
    else:
        st.info("Run forecasting/model_comparison to generate model_comparison.parquet.")

elif page == "Backtest Results":
    st.header("Backtest Results")
    returns_dir = Path("data/processed/returns")
    if (returns_dir / "backtest_results.parquet").exists():
        import pandas as pd
        df = pd.read_parquet(returns_dir / "backtest_results.parquet")
        st.metric("Annualised Sharpe", df["annualised_sharpe"].iloc[0] if "annualised_sharpe" in df.columns else "—")
        st.metric("Annualised L/S Return", df["annualised_return_ls"].iloc[0] if "annualised_return_ls" in df.columns else "—")
        st.dataframe(df)
    else:
        st.info("Run backtesting/portfolio_simulator to generate backtest_results.parquet.")
