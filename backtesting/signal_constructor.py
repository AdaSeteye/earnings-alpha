"""
Long/short signal from model scores (quintiles).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_config() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "data_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def construct_signal(
    predictions: pd.DataFrame,
    signal_col: str = "tft_pred_20d",
    long_quantile: float = 0.8,
    short_quantile: float = 0.2,
) -> pd.Series:
    """Return signal: +1 long, -1 short, 0 flat by quantile."""
    s = predictions[signal_col]
    long_thresh = s.quantile(long_quantile)
    short_thresh = s.quantile(short_quantile)
    out = pd.Series(0, index=predictions.index)
    out[s >= long_thresh] = 1
    out[s <= short_thresh] = -1
    return out


if __name__ == "__main__":
    config = load_config()
    paths = config.get("paths", {})
    feat_path = Path(paths.get("processed_features", "data/processed/features")) / "forecast_feature_matrix.parquet"
    if feat_path.exists():
        df = pd.read_parquet(feat_path)
        if "xgb_pred_20d" in df.columns:
            df["signal"] = construct_signal(df.rename(columns={"xgb_pred_20d": "tft_pred_20d"}), signal_col="tft_pred_20d")
            print(df[["call_id", "signal"]].head())
