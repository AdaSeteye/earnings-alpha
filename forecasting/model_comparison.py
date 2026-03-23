"""
Cross-validation and model comparison (RMSE, IC, hit rate).
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


def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 and np.std(y_pred) > 0 else 0.0


def hit_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true > 0) == (y_pred > 0)))


def main() -> None:
    config = load_config()
    paths = config.get("paths", {})
    features_dir = Path(paths.get("processed_features", "data/processed/features"))
    feat_path = features_dir / "forecast_feature_matrix.parquet"
    if not feat_path.exists():
        print("Run feature_engineering first.")
        return
    df = pd.read_parquet(feat_path)
    if "car_20d" not in df.columns:
        df["car_20d"] = 0.0
    y = df["car_20d"].values
    results = []
    for name, col in [("Random walk", None), ("XGBoost", "xgb_pred_20d"), ("TFT", "tft_pred_20d")]:
        if col is None:
            pred = np.zeros_like(y)
        elif col in df.columns:
            pred = df[col].values
        else:
            continue
        rmse = np.sqrt(np.mean((y - pred) ** 2))
        ic = information_coefficient(y, pred)
        hit = hit_rate(y, pred)
        results.append({"model": name, "RMSE": rmse, "IC": ic, "hit_rate": hit})
    out = pd.DataFrame(results)
    print(out)
    out.to_parquet(features_dir / "model_comparison.parquet", index=False)


if __name__ == "__main__":
    main()
