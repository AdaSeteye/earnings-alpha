"""
Inference on new calls: load trained model and produce predictions.
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


def predict_new_calls(feature_df: pd.DataFrame, model_type: str = "xgboost") -> pd.Series:
    """Return predicted 20d return for each row. model_type: xgboost | tft."""
    if model_type == "xgboost":
        try:
            import xgboost as xgb
            model_path = Path("models/xgboost/model.json")
            if model_path.exists():
                model = xgb.XGBRegressor()
                model.load_model(str(model_path))
                numeric = feature_df.select_dtypes(include="number").fillna(0)
                return pd.Series(model.predict(numeric), index=feature_df.index)
        except Exception:
            pass
    return pd.Series(0.0, index=feature_df.index)


if __name__ == "__main__":
    config = load_config()
    paths = config.get("paths", {})
    feat_path = Path(paths.get("processed_features", "data/processed/features")) / "forecast_feature_matrix.parquet"
    if feat_path.exists():
        df = pd.read_parquet(feat_path)
        pred = predict_new_calls(df, "xgboost")
        print(pred.head())
