"""
Rule-based and model-based uncertainty scoring per speaker turn.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm

from .hedging_lexicon import count_uncertainty_ratio, get_hedging_phrases


def load_config() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "data_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def score_hedging_phrases(text: str) -> float:
    """Fraction of hedging phrases found in text (normalised by word count)."""
    text_lower = text.lower()
    words = text_lower.split()
    if not words:
        return 0.0
    count = sum(1 for p in get_hedging_phrases() if p in text_lower)
    return min(1.0, count / max(1, len(words) / 50))


def main() -> None:
    config = load_config()
    paths = config.get("paths", {})
    parsed_dir = Path(paths.get("processed_transcripts", "data/processed/transcripts_parsed"))
    features_dir = Path(paths.get("processed_features", "data/processed/features"))
    features_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = list(parsed_dir.glob("*_parsed.parquet"))
    if not parquet_files:
        print("No parsed transcripts. Run transcript_parser first.")
        return

    rows = []
    for path in tqdm(parquet_files, desc="Uncertainty scoring"):
        df = pd.read_parquet(path)
        if "text" not in df.columns:
            continue
        for _, row in df.iterrows():
            text = str(row["text"])
            u_ratio = count_uncertainty_ratio(text)
            h_ratio = score_hedging_phrases(text)
            rows.append({
                **{k: v for k, v in row.items()},
                "uncertainty_ratio": u_ratio,
                "hedging_ratio": h_ratio,
                "uncertainty_score": (u_ratio + h_ratio) / 2,
            })
    if not rows:
        print("No turns to save.")
        return
    out_df = pd.DataFrame(rows)
    out_path = features_dir / "uncertainty_turns.parquet"
    out_df.to_parquet(out_path, index=False)
    print(f"Saved {len(out_df)} turn-level uncertainty scores to {out_path}")


if __name__ == "__main__":
    main()
