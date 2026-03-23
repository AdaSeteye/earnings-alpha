"""
Feature construction for forecasting models: NLP, price, earnings, macro, static.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_config() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "data_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_feature_matrix(
    sentiment_path: Path,
    uncertainty_path: Path,
    topic_path: Path,
    price_path: Path,
    returns_path: Path | None = None,
) -> pd.DataFrame:
    """Merge per-call NLP features with price/return data into one feature matrix."""
    dfs = []
    if sentiment_path.exists():
        dfs.append(pd.read_parquet(sentiment_path))
    if uncertainty_path.exists():
        unc = pd.read_parquet(uncertainty_path)
        if dfs:
            dfs[0] = dfs[0].merge(unc, on="call_id", how="left", suffixes=("", "_unc"))
        else:
            dfs.append(unc)
    if topic_path.exists():
        top = pd.read_parquet(topic_path)
        if dfs:
            dfs[0] = dfs[0].merge(top, on="call_id", how="left")
        else:
            dfs.append(top)
    if not dfs:
        return pd.DataFrame()
    feats = dfs[0]
    for d in dfs[1:]:
        if "call_id" in d.columns and d is not dfs[0]:
            feats = feats.merge(d, on="call_id", how="left")
    feats["ticker"] = feats["call_id"].str.split("_").str[0]
    feats["event_date"] = feats["call_id"].str.split("_").str[1]
    if price_path.exists():
        prices = pd.read_parquet(price_path)
        prices["Date"] = pd.to_datetime(prices["Date"] if "Date" in prices.columns else prices.get("index", prices.index))
        # Add momentum/volatility per ticker would go here
    return feats


def main() -> None:
    config = load_config()
    paths = config.get("paths", {})
    features_dir = Path(paths.get("processed_features", "data/processed/features"))
    raw_prices = Path(paths.get("raw_prices", "data/raw/prices")) / "daily_prices.parquet"
    returns_dir = Path(paths.get("processed_returns", "data/processed/returns"))
    df = build_feature_matrix(
        features_dir / "sentiment_per_call.parquet",
        features_dir / "uncertainty_per_call.parquet",
        features_dir / "topic_per_call.parquet",
        raw_prices,
        returns_dir / "car_event_study.parquet" if (returns_dir / "car_event_study.parquet").exists() else None,
    )
    if df.empty:
        print("No feature files found. Run NLP pipelines first.")
        return
    out_path = features_dir / "forecast_feature_matrix.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved feature matrix to {out_path}")


if __name__ == "__main__":
    main()
