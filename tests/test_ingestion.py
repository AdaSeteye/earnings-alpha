"""Tests for ingestion module."""
from pathlib import Path

import pytest


def test_config_paths():
    from ingestion.transcript_parser import load_config
    config = load_config()
    assert "paths" in config
    assert "raw_transcripts" in config.get("paths", {})


def test_parse_text_transcript():
    from ingestion.transcript_parser import parse_text_transcript
    text = "Operator: Good day.\nJohn Smith, CEO: Thank you. Our results were strong.\nAnalyst: What about guidance?"
    turns = parse_text_transcript(text)
    assert len(turns) >= 1
    assert all("speaker" in t and "text" in t for t in turns)


def test_hedging_lexicon():
    from nlp.uncertainty.hedging_lexicon import count_uncertainty_ratio, get_uncertainty_lexicon
    assert len(get_uncertainty_lexicon()) > 0
    r = count_uncertainty_ratio("We believe it may be possible that growth could continue.")
    assert 0 <= r <= 1
