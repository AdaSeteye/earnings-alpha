"""
Position sizing, rebalancing, long/short portfolio simulation.
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
    parser.add_argument("--config", default="configs/backtest_config.yaml")
    parser.add_argument("--signal-col", default="tft_pred_20d")
    parser.add_argument("--long-quantile", type=float, default=0.8)
    parser.add_argument("--short-quantile", type=float, default=0.2)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    args = parser.parse_args()
    with open(Path(args.config)) as f:
        bt_cfg = yaml.safe_load(f)
    signal_cfg = bt_cfg.get("signal", {})
    signal_col = args.signal_col or signal_cfg.get("signal_col", "tft_pred_20d")
    long_q = args.long_quantile or signal_cfg.get("long_quantile", 0.8)
    short_q = args.short_quantile or signal_cfg.get("short_quantile", 0.2)
    cost_bps = args.transaction_cost_bps or bt_cfg.get("costs", {}).get("transaction_cost_bps", 5)
    data_cfg = load_config()
    paths = data_cfg.get("paths", {})
    features_dir = Path(paths.get("processed_features", "data/processed/features"))
    returns_dir = Path(paths.get("processed_returns", "data/processed/returns"))
    returns_dir.mkdir(parents=True, exist_ok=True)
    feat_path = features_dir / "forecast_feature_matrix.parquet"
    if not feat_path.exists():
        print("Run feature_engineering first.")
        return
    df = pd.read_parquet(feat_path)
    pred_col = signal_col if signal_col in df.columns else "xgb_pred_20d" if "xgb_pred_20d" in df.columns else None
    if pred_col is None:
        df["signal_value"] = 0.0
    else:
        s = df[pred_col]
        long_thresh = s.quantile(long_q)
        short_thresh = s.quantile(short_q)
        df["signal_value"] = 0.0
        df.loc[s >= long_thresh, "signal_value"] = 1.0
        df.loc[s <= short_thresh, "signal_value"] = -1.0
    car_col = "car_20d" if "car_20d" in df.columns else None
    if car_col:
        strat_ret = (df["signal_value"] * df[car_col]).mean()
        n_longs = (df["signal_value"] == 1).sum()
        n_shorts = (df["signal_value"] == -1).sum()
        sharpe = strat_ret / (df[car_col].std() + 1e-12) * np.sqrt(252 / 20) if df[car_col].std() > 0 else 0
        results = {
            "annualised_sharpe": float(sharpe),
            "annualised_return_ls": float(strat_ret * (252 / 20)),
            "n_long": int(n_longs),
            "n_short": int(n_shorts),
            "transaction_cost_bps": cost_bps,
        }
    else:
        results = {"annualised_sharpe": 0.0, "annualised_return_ls": 0.0, "n_long": 0, "n_short": 0}
    pd.DataFrame([results]).to_parquet(returns_dir / "backtest_results.parquet", index=False)
    print("Backtest results:", results)


if __name__ == "__main__":
    main()
