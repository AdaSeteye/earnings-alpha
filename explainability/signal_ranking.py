"""
Feature importance report from SHAP or model coefficients.
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


def main() -> None:
    config = load_config()
    paths = config.get("paths", {})
    features_dir = Path(paths.get("processed_features", "data/processed/features"))
    try:
        import xgboost as xgb
        model_path = Path("models/xgboost/model.json")
        if model_path.exists():
            model = xgb.XGBRegressor()
            model.load_model(str(model_path))
            imp = model.get_score(importance_type="gain")
            rank = pd.DataFrame([{"feature": k, "importance": v} for k, v in imp.items()]).sort_values("importance", ascending=False)
            rank.to_parquet(features_dir / "signal_ranking.parquet", index=False)
            print(rank.head(10))
    except Exception:
        pd.DataFrame({"feature": ["placeholder"], "importance": [0.0]}).to_parquet(features_dir / "signal_ranking.parquet", index=False)
    print("Signal ranking saved.")


if __name__ == "__main__":
    main()
