"""Backtest results page."""
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Backtest Results", layout="wide")
st.header("Portfolio Performance")
returns_dir = Path("data/processed/returns")
if (returns_dir / "backtest_results.parquet").exists():
    df = __import__("pandas").read_parquet(returns_dir / "backtest_results.parquet")
    st.dataframe(df)
    for col in ["annualised_sharpe", "annualised_return_ls"]:
        if col in df.columns:
            st.metric(col.replace("_", " ").title(), f"{df[col].iloc[0]:.4f}")
