"""Causal estimates page."""
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Causal Estimates", layout="wide")
st.header("DiD / Event Study Results")
returns_dir = Path("data/processed/returns")
if (returns_dir / "did_estimates.parquet").exists():
    df = __import__("pandas").read_parquet(returns_dir / "did_estimates.parquet")
    st.dataframe(df)
if (returns_dir / "car_event_study.parquet").exists():
    df = __import__("pandas").read_parquet(returns_dir / "car_event_study.parquet")
    st.subheader("CAR Event Study")
    st.dataframe(df)
