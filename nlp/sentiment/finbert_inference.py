"""
Batch FinBERT inference at speaker-turn level; writes sentiment scores per turn.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm


def load_config() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "data_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model_config() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "model_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_inference(texts: list[str], model_path: str | None, cfg: dict) -> list[dict[str, float]]:
    """Run FinBERT on a list of texts; return list of {positive, negative, neutral} scores."""
    model_name = cfg.get("base_model", "ProsusAI/finbert")
    max_length = cfg.get("max_length", 256)
    batch_size = cfg.get("batch_size", 32)
    try:
        from transformers import pipeline
        pipe = pipeline(
            "sentiment-analysis",
            model=model_path or model_name,
            tokenizer=model_path or model_name,
            device=-1,
            truncation=True,
            max_length=max_length,
        )
        # FinBERT-style labels
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            out = pipe(batch)
            for item in out:
                label = item.get("label", "neutral").upper()
                score = item.get("score", 0.0)
                res = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
                if "POS" in label or "POSITIVE" in label:
                    res["positive"] = score
                elif "NEG" in label or "NEGATIVE" in label:
                    res["negative"] = score
                else:
                    res["neutral"] = score
                results.append(res)
        return results
    except Exception as e:
        print(f"Inference error: {e}. Returning neutral scores.")
        return [{"positive": 0.0, "negative": 0.0, "neutral": 1.0} for _ in texts]


def main() -> None:
    data_cfg = load_config()
    model_cfg = load_model_config().get("finbert", {})
    paths = data_cfg.get("paths", {})
    parsed_dir = Path(paths.get("processed_transcripts", "data/processed/transcripts_parsed"))
    features_dir = Path(paths.get("processed_features", "data/processed/features"))
    features_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path("models/finbert/finetuned")
    use_finetuned = model_path.exists()
    model_path = str(model_path) if use_finetuned else None

    parquet_files = list(parsed_dir.glob("*_parsed.parquet"))
    if not parquet_files:
        print(f"No parsed transcripts in {parsed_dir}. Run transcript_parser first.")
        return

    all_turns = []
    for path in tqdm(parquet_files, desc="Sentiment inference"):
        df = pd.read_parquet(path)
        if df.empty or "text" not in df.columns:
            continue
        texts = df["text"].astype(str).tolist()
        scores = run_inference(texts, model_path, model_cfg)
        for i, row in df.iterrows():
            s = scores[i] if i < len(scores) else {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
            all_turns.append({
                **row.to_dict(),
                "sent_positive": s["positive"],
                "sent_negative": s["negative"],
                "sent_neutral": s["neutral"],
                "sent_score": s["positive"] - s["negative"],
            })
    if not all_turns:
        print("No turns to save.")
        return
    out_df = pd.DataFrame(all_turns)
    out_path = features_dir / "sentiment_turns.parquet"
    out_df.to_parquet(out_path, index=False)
    print(f"Saved {len(out_df)} turn-level sentiment scores to {out_path}")


if __name__ == "__main__":
    main()
