"""
Per-call uncertainty aggregation: overall ratio, CEO uncertainty, change vs prior call, Q&A spike.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_config() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "data_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def extract_call_id(source_file: str) -> str:
    parts = source_file.replace("_parsed", "").split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return source_file


def main() -> None:
    config = load_config()
    paths = config.get("paths", {})
    features_dir = Path(paths.get("processed_features", "data/processed/features"))
    path = features_dir / "uncertainty_turns.parquet"
    if not path.exists():
        print("Run uncertainty_scorer first.")
        return
    df = pd.read_parquet(path)
    df["call_id"] = df["source_file"].map(extract_call_id)
    overall = df.groupby("call_id")["uncertainty_score"].mean().reset_index()
    overall.columns = ["call_id", "uncertainty_overall"]
    ceo = df[df["role"] == "CEO"].groupby("call_id")["uncertainty_score"].mean().reset_index()
    ceo.columns = ["call_id", "uncertainty_ceo"]
    qa = df[df["section"] == "qa"].groupby("call_id")["uncertainty_score"].mean().reset_index()
    qa.columns = ["call_id", "uncertainty_qa_mean"]
    prep = df[df["section"] == "prepared_remarks"].groupby("call_id")["uncertainty_score"].mean().reset_index()
    prep.columns = ["call_id", "uncertainty_prepared_mean"]
    agg = overall.merge(ceo, on="call_id", how="left").merge(qa, on="call_id", how="left").merge(prep, on="call_id", how="left")
    agg["uncertainty_qa_spike"] = agg["uncertainty_qa_mean"] - agg["uncertainty_prepared_mean"]
    agg["uncertainty_change_prior"] = np.nan  # would need time-ordered calls per ticker
    out_path = features_dir / "uncertainty_per_call.parquet"
    agg.to_parquet(out_path, index=False)
    print(f"Saved per-call uncertainty features to {out_path}")


if __name__ == "__main__":
    main()
