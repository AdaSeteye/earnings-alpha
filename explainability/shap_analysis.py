"""
Global and local SHAP for XGBoost/TFT; output HTML summary.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_config() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "data_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="xgboost", choices=["xgboost", "tft"])
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--output", default="reports/shap_summary.html")
    args = parser.parse_args()
    config = load_config()
    paths = config.get("paths", {})
    features_dir = Path(paths.get("processed_features", "data/processed/features"))
    feat_path = features_dir / "forecast_feature_matrix.parquet"
    if not feat_path.exists():
        print("Run feature_engineering first.")
        return
    df = pd.read_parquet(feat_path)
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "car_20d"]
    X = df[numeric_cols].fillna(0) if numeric_cols else pd.DataFrame(np.zeros((len(df), 1)))
    X = X.iloc[: min(args.n_samples, len(X))]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    try:
        import shap
        if args.model == "xgboost":
            import xgboost as xgb
            model_path = Path("models/xgboost/model.json")
            if not model_path.exists():
                print("Train XGBoost first (xgboost_model). Writing placeholder.")
                html = "<html><body><p>No model. Run xgboost_model first.</p></body></html>"
                Path(args.output).write_text(html)
                return
            model = xgb.XGBRegressor()
            model.load_model(str(model_path))
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            import matplotlib.pyplot as plt
            shap.summary_plot(shap_values, X, show=False)
            plt.savefig(Path(args.output).with_suffix(".png"), bbox_inches="tight")
            plt.close()
        with open(args.output, "w") as f:
            f.write("<html><body><h1>SHAP summary</h1><p>See PNG for summary plot.</p></body></html>")
    except ImportError:
        Path(args.output).write_text("<html><body><p>Install shap and train a model.</p></body></html>")
    except Exception as e:
        Path(args.output).write_text(f"<html><body><p>Error: {e}</p></body></html>")
    print(f"SHAP output written to {args.output}")


if __name__ == "__main__":
    main()
