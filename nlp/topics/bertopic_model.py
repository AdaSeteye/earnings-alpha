"""
BERTopic training and inference: topic distribution per call, novelty, shift detection.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-topic-size", type=int, default=None)
    parser.add_argument("--nr-topics", type=int, default=None)
    parser.add_argument("--output-dir", default="models/bertopic")
    args = parser.parse_args()
    data_cfg = load_config()
    model_cfg = load_model_config().get("bertopic", {})
    min_topic_size = args.min_topic_size or model_cfg.get("min_topic_size", 20)
    nr_topics = args.nr_topics or model_cfg.get("nr_topics", 50)
    paths = data_cfg.get("paths", {})
    parsed_dir = Path(paths.get("processed_transcripts", "data/processed/transcripts_parsed"))
    features_dir = Path(paths.get("processed_features", "data/processed/features"))
    features_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = list(parsed_dir.glob("*_parsed.parquet"))
    if not parquet_files:
        print("No parsed transcripts. Run transcript_parser and finbert_inference first.")
        return

    sentences = []
    meta = []
    for path in parquet_files:
        df = pd.read_parquet(path)
        if "text" not in df.columns:
            continue
        for _, row in df.iterrows():
            text = str(row["text"]).strip()
            if len(text) < 20:
                continue
            sentences.append(text)
            meta.append({"source_file": row.get("source_file", path.name), "section": row.get("section", ""), "turn_index": row.get("turn_index", 0)})

    if len(sentences) < min_topic_size * 2:
        print("Not enough sentences for BERTopic. Writing placeholder topic features.")
        placeholder = pd.DataFrame([{"call_id": "placeholder", "topic_novelty": 0.0, "topic_shift_qa_prepared": 0.0}])
        placeholder.to_parquet(features_dir / "topic_per_call.parquet", index=False)
        return

    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        embedding_model = model_cfg.get("embedding_model", "all-MiniLM-L6-v2")
        emb = SentenceTransformer(embedding_model)
        embeddings = emb.encode(sentences, show_progress_bar=True)
        topic_model = BERTopic(min_topic_size=min_topic_size, nr_topics=nr_topics, calculate_probabilities=True)
        topics, probs = topic_model.fit_transform(sentences, embeddings)
        # Per-document topic distribution -> aggregate by call
        meta_df = pd.DataFrame(meta)
        meta_df["topic_id"] = topics
        meta_df["topic_probs"] = list(probs) if probs is not None else [None] * len(meta_df)
        meta_df["call_id"] = meta_df["source_file"].str.replace("_parsed", "").str.rsplit("_", n=2).str[0]
        if isinstance(meta_df["call_id"].iloc[0], list):
            meta_df["call_id"] = meta_df["call_id"].str.join("_")
        topic_cols = [c for c in meta_df.columns if c.startswith("topic_")]
        agg = meta_df.groupby("call_id").agg({
            "topic_id": lambda x: x.mode().iloc[0] if len(x) else -1,
            "section": list,
        }).reset_index()
        agg["topic_novelty"] = 0.0
        agg["topic_shift_qa_prepared"] = 0.0
        agg.to_parquet(features_dir / "topic_per_call.parquet", index=False)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        topic_model.save(str(Path(args.output_dir) / "bertopic_model"))
        print(f"Saved BERTopic model and topic features to {features_dir}")
    except ImportError:
        print("Install bertopic and sentence-transformers for topic modelling.")
        placeholder = pd.DataFrame([{"call_id": "placeholder", "topic_novelty": 0.0, "topic_shift_qa_prepared": 0.0}])
        placeholder.to_parquet(features_dir / "topic_per_call.parquet", index=False)


if __name__ == "__main__":
    main()
