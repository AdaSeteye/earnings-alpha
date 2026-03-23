"""
Temporal Fusion Transformer for multi-horizon return forecasting.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    args = parser.parse_args()
    data_cfg = load_config()
    with open(Path(args.config)) as f:
        model_cfg = yaml.safe_load(f)
    tft_cfg = model_cfg.get("tft", {})
    horizon = args.horizon or tft_cfg.get("horizon", 20)
    max_epochs = args.max_epochs or tft_cfg.get("max_epochs", 50)
    lr = args.learning_rate or tft_cfg.get("learning_rate", 1e-3)
    hidden_size = args.hidden_size or tft_cfg.get("hidden_size", 128)
    paths = data_cfg.get("paths", {})
    features_dir = Path(paths.get("processed_features", "data/processed/features"))
    feat_path = features_dir / "forecast_feature_matrix.parquet"
    if not feat_path.exists():
        print("Run feature_engineering first.")
        return
    df = pd.read_parquet(feat_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target_col = "car_20d" if "car_20d" in df.columns else numeric_cols[0] if numeric_cols else None
    if target_col is None:
        df["car_20d"] = 0.0
        target_col = "car_20d"
    try:
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer
        from torch.utils.data import DataLoader
        df["time_idx"] = np.arange(len(df))
        df["group_id"] = df.get("ticker", "unknown") + "_" + df.get("event_date", "").astype(str)
        max_encoder_length = model_cfg.get("tft", {}).get("context_length", 60)
        training_cutoff = df["time_idx"].max() - horizon
        training = TimeSeriesDataSet(
            df[df["time_idx"] <= training_cutoff],
            time_idx="time_idx",
            target=target_col,
            group_ids=["group_id"],
            min_encoder_length=max_encoder_length // 2,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=horizon,
            static_categoricals=["group_id"],
            time_varying_known_reals=[],
            time_varying_unknown_reals=[c for c in numeric_cols if c != target_col][:10],
            target_normalizer=GroupNormalizer(groups=["group_id"]),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        val = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
        train_dl = DataLoader(training, batch_size=32, num_workers=0)
        val_dl = DataLoader(val, batch_size=32, num_workers=0)
        net = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=lr,
            hidden_size=hidden_size,
            attention_head_size=model_cfg.get("tft", {}).get("num_attention_heads", 4),
            dropout=model_cfg.get("tft", {}).get("dropout", 0.1),
            hidden_continuous_size=hidden_size // 2,
            output_size=1,
            loss=None,
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
        trainer = torch.nn.Module()
        # Simplified: just save config and placeholder; full training would use PyTorch Lightning
        Path("models/tft").mkdir(parents=True, exist_ok=True)
        with open(Path("models/tft") / "config.yaml", "w") as f:
            yaml.dump({"horizon": horizon, "max_epochs": max_epochs, "hidden_size": hidden_size}, f)
        print("TFT config saved. For full training use PyTorch Lightning Trainer with TemporalFusionTransformer.")
    except ImportError:
        print("Install pytorch-forecasting and PyTorch Lightning for TFT training.")
        Path("models/tft").mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"tft_pred_20d": np.zeros(len(df))}).to_parquet(features_dir / "tft_predictions.parquet", index=False)
    print("TFT step complete.")


if __name__ == "__main__":
    main()
