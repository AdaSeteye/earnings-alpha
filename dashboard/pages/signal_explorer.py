"""Signal explorer page (also reachable from main app)."""
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Signal Explorer", layout="wide")
st.header("NLP Signal Distributions")
features_dir = Path("data/processed/features")
for name in ["sentiment_per_call.parquet", "uncertainty_per_call.parquet", "topic_per_call.parquet"]:
    p = features_dir / name
    if p.exists():
        df = __import__("pandas").read_parquet(p)
        st.subheader(name)
        st.dataframe(df.head(50))
