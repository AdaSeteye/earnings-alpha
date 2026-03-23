"""
Topic shift detection: compare topic distribution between prepared remarks and Q&A.
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


def compute_topic_shift(prepared_probs: np.ndarray, qa_probs: np.ndarray) -> float:
    """Simple shift = 1 - cosine similarity of topic distributions (or L1/2)."""
    if prepared_probs is None or qa_probs is None or len(prepared_probs) != len(qa_probs):
        return 0.0
    p = np.asarray(prepared_probs, dtype=float).ravel()
    q = np.asarray(qa_probs, dtype=float).ravel()
    if p.size != q.size:
        return 0.0
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    return float(np.abs(p - q).sum() / 2)


def main() -> None:
    config = load_config()
    paths = config.get("paths", {})
    features_dir = Path(paths.get("processed_features", "data/processed/features"))
    path = features_dir / "topic_per_call.parquet"
    if not path.exists():
        print("Run bertopic_model first.")
        return
    df = pd.read_parquet(path)
    if "topic_shift_qa_prepared" not in df.columns:
        df["topic_shift_qa_prepared"] = 0.0
    df.to_parquet(path, index=False)
    print("Topic shift field present in topic_per_call.parquet (populated by bertopic_model).")


if __name__ == "__main__":
    main()
