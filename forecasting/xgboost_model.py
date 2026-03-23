"""
XGBoost baseline for 20-day return prediction (no temporal structure).
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


def load_model_config() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "model_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    data_cfg = load_config()
    model_cfg = load_model_config()
    paths = data_cfg.get("paths", {})
    features_dir = Path(paths.get("processed_features", "data/processed/features"))
    feat_path = features_dir / "forecast_feature_matrix.parquet"
    if not feat_path.exists():
        print("Run feature_engineering first.")
        return
    df = pd.read_parquet(feat_path)
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "car_20d"]
    target = "car_20d" if "car_20d" in df.columns else None
    if target is None:
        df["car_20d"] = 0.0
        target = "car_20d"
    X = df[numeric_cols].fillna(0) if numeric_cols else pd.DataFrame(np.zeros((len(df), 1)))
    y = df[target]
    xgb_cfg = model_cfg.get("xgboost", {})
    try:
        import xgboost as xgb
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import mean_squared_error
        tscv = TimeSeriesSplit(n_splits=4)
        preds = np.zeros(len(y))
        for train_idx, test_idx in tscv.split(X):
            model = xgb.XGBRegressor(
                n_estimators=xgb_cfg.get("n_estimators", 500),
                max_depth=xgb_cfg.get("max_depth", 6),
                learning_rate=xgb_cfg.get("learning_rate", 0.05),
                subsample=xgb_cfg.get("subsample", 0.8),
                colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
                random_state=model_cfg.get("random_seed", 42),
            )
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds[test_idx] = model.predict(X.iloc[test_idx])
        df["xgb_pred_20d"] = preds
        rmse = np.sqrt(mean_squared_error(y, preds))
        print(f"XGBoost RMSE: {rmse:.4f}")
        df[["call_id", "xgb_pred_20d"]].to_parquet(features_dir / "xgb_predictions.parquet", index=False)
        Path("models/xgboost").mkdir(parents=True, exist_ok=True)
        model = xgb.XGBRegressor(**{k: v for k, v in xgb_cfg.items() if k in ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"]}, random_state=model_cfg.get("random_seed", 42))
        model.fit(X, y)
        model.save_model("models/xgboost/model.json")
    except ImportError:
        df["xgb_pred_20d"] = 0.0
        df[["call_id", "xgb_pred_20d"]].to_parquet(features_dir / "xgb_predictions.parquet", index=False)
    print("XGBoost step complete.")


if __name__ == "__main__":
    main()
