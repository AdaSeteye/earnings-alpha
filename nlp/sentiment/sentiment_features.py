"""
Per-call aggregation of sentiment: mean by speaker, trajectory slope, CEO-CFO gap, Q&A vs prepared.
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
    """Derive call identifier from parsed filename (e.g. AAPL_2024-02-01_... -> AAPL_2024-02-01)."""
    parts = source_file.replace("_parsed", "").split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return source_file


def aggregate_sentiment(turns_df: pd.DataFrame) -> pd.DataFrame:
    """One row per call: mean sentiment by role, trajectory slope, CEO-CFO gap, Q&A vs prepared."""
    turns_df = turns_df.copy()
    turns_df["call_id"] = turns_df["source_file"].map(extract_call_id)
    rows = []
    for call_id, grp in turns_df.groupby("call_id"):
        grp = grp.sort_values("turn_index")
        row = {"call_id": call_id}
        for role in ["CEO", "CFO", "Analyst", "Other"]:
            sub = grp[grp["role"] == role]
            if not sub.empty and "sent_score" in sub.columns:
                row[f"sent_mean_{role}"] = sub["sent_score"].mean()
            else:
                row[f"sent_mean_{role}"] = np.nan
        if "sent_score" in grp.columns and len(grp) >= 2:
            x = np.arange(len(grp))
            slope = np.polyfit(x, grp["sent_score"].values, 1)[0]
            row["sent_trajectory_slope"] = slope
        else:
            row["sent_trajectory_slope"] = np.nan
        row["sent_ceo_cfo_gap"] = row.get("sent_mean_CEO", np.nan) - row.get("sent_mean_CFO", np.nan)
        prep = grp[grp["section"] == "prepared_remarks"]
        qa = grp[grp["section"] == "qa"]
        row["sent_prepared_mean"] = prep["sent_score"].mean() if "sent_score" in prep.columns and not prep.empty else np.nan
        row["sent_qa_mean"] = qa["sent_score"].mean() if "sent_score" in qa.columns and not qa.empty else np.nan
        row["sent_qa_prepared_divergence"] = (row["sent_qa_mean"] - row["sent_prepared_mean"]) if not (np.isnan(row["sent_qa_mean"]) or np.isnan(row["sent_prepared_mean"])) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    config = load_config()
    paths = config.get("paths", {})
    features_dir = Path(paths.get("processed_features", "data/processed/features"))
    sentiment_path = features_dir / "sentiment_turns.parquet"
    if not sentiment_path.exists():
        print("Run finbert_inference first to create sentiment_turns.parquet")
        return
    df = pd.read_parquet(sentiment_path)
    agg = aggregate_sentiment(df)
    out_path = features_dir / "sentiment_per_call.parquet"
    agg.to_parquet(out_path, index=False)
    print(f"Saved per-call sentiment features to {out_path}")


if __name__ == "__main__":
    main()
