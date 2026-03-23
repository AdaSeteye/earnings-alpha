"""Model performance page."""
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Model Performance", layout="wide")
st.header("Forecast Accuracy")
features_dir = Path("data/processed/features")
if (features_dir / "model_comparison.parquet").exists():
    df = __import__("pandas").read_parquet(features_dir / "model_comparison.parquet")
    st.dataframe(df)
    st.bar_chart(df.set_index("model"))
