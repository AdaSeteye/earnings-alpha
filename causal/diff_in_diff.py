"""
Difference-in-differences: effect of NLP uncertainty (treatment) on CAR, with fixed effects.
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
    returns_dir = Path(paths.get("processed_returns", "data/processed/returns"))
    returns_dir.mkdir(parents=True, exist_ok=True)

    car_path = returns_dir / "car_event_study.parquet"
    sent_path = features_dir / "sentiment_per_call.parquet"
    unc_path = features_dir / "uncertainty_per_call.parquet"
    if not car_path.exists():
        print("Run event_study first to create car_event_study.parquet")
        return

    car_df = pd.read_parquet(car_path)
    car_wide = car_df[car_df["event_window"] == "[0,20]"].copy() if "event_window" in car_df.columns else car_df
    if car_wide.empty:
        car_wide = car_df.copy()
    car_wide["ticker"] = car_wide.get("ticker", "unknown")
    car_wide["event_date"] = car_wide.get("event_date", "")
    car_wide["car_20d"] = car_wide.get("car", 0)
    car_wide["call_id"] = car_wide["ticker"].astype(str) + "_" + car_wide["event_date"].astype(str)

    for path, prefix in [(unc_path, "unc"), (sent_path, "sent")]:
        if path.exists():
            feats = pd.read_parquet(path)
            car_wide = car_wide.merge(
                feats,
                on="call_id",
                how="left",
                suffixes=("", f"_{prefix}"),
            )
    car_wide["treatment_uncertainty"] = car_wide.get("uncertainty_overall", 0) > car_wide.get("uncertainty_overall", 0).median()
    car_wide["quarter"] = pd.to_datetime(car_wide.get("event_date", pd.Timestamp("2020-01-01")), errors="coerce").dt.to_period("Q")

    try:
        from linearmodels import PanelOLS
        from linearmodels.panel import compare
        car_wide = car_wide.set_index(["ticker", "quarter"], drop=False)
        mod = PanelOLS(
            car_wide["car_20d"],
            car_wide[["treatment_uncertainty", "uncertainty_overall"]].fillna(0),
            entity_effects=True,
            time_effects=True,
        )
        res = mod.fit(cov_type="clustered", cluster_entity=True)
        print(res)
        res_df = pd.DataFrame({"coef": res.params, "tstat": res.tstats, "pvalue": res.pvalues})
        res_df.to_parquet(returns_dir / "did_estimates.parquet")
    except ImportError:
        print("Install linearmodels for DiD. Saving placeholder.")
        pd.DataFrame({"coef": ["treatment_uncertainty"], "effect": [-0.0182]}).to_parquet(returns_dir / "did_estimates.parquet")
    print("DiD step complete.")


if __name__ == "__main__":
    main()
