"""Tests for NLP pipeline."""
import numpy as np
import pandas as pd


def test_sentiment_aggregation():
    from nlp.sentiment.sentiment_features import aggregate_sentiment, extract_call_id
    assert extract_call_id("AAPL_2024-02-01_parsed.parquet") == "AAPL_2024-02-01"
    turns = pd.DataFrame({
        "call_id": ["c1", "c1", "c1"],
        "source_file": ["c1_parsed", "c1_parsed", "c1_parsed"],
        "role": ["CEO", "CFO", "Analyst"],
        "section": ["prepared_remarks", "prepared_remarks", "qa"],
        "turn_index": [0, 1, 2],
        "sent_score": [0.1, -0.1, 0.0],
    })
    agg = aggregate_sentiment(turns)
    assert len(agg) == 1
    assert "sent_ceo_cfo_gap" in agg.columns
