"""
Per-call topic features: distribution vector, novelty, guidance coverage (stub).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_config() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "data_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    config = load_config()
    paths = config.get("paths", {})
    features_dir = Path(paths.get("processed_features", "data/processed/features"))
    path = features_dir / "topic_per_call.parquet"
    if not path.exists():
        print("Run bertopic_model first.")
        return
    df = pd.read_parquet(path)
    if "guidance_topic_coverage" not in df.columns:
        df["guidance_topic_coverage"] = 0.0
    df.to_parquet(path, index=False)
    print("Topic features ready in topic_per_call.parquet.")


if __name__ == "__main__":
    main()
