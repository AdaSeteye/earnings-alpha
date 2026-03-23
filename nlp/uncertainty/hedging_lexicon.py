"""
Loughran-McDonald style uncertainty word list and hedging phrases.
"""
from __future__ import annotations

# Subset of uncertainty-related words (Loughran-McDonald extended)
UNCERTAINTY_WORDS = {
    "approximately", "assume", "assumption", "believe", "could", "depends", "estimate",
    "estimated", "estimates", "expect", "expected", "expects", "may", "might", "possible",
    "possibly", "potential", "potentially", "predict", "predicted", "predictions",
    "probably", "risk", "risks", "uncertain", "uncertainty", "unclear", "unlikely",
    "subject to", "conditional", "contingent", "doubt", "doubtful", "perhaps",
    "likely", "likelihood", "possibility", "probable", "speculative", "vague",
}

HEDGING_PHRASES = [
    "we believe", "we expect", "we think", "it is possible", "it is likely",
    "subject to", "depending on", "to the extent", "may not", "might not",
    "could be", "would be", "generally", "typically", "often", "sometimes",
]

MODAL_VERBS = {"may", "might", "could", "would", "should", "can", "must"}


def get_uncertainty_lexicon() -> set[str]:
    return UNCERTAINTY_WORDS.copy()


def get_hedging_phrases() -> list[str]:
    return HEDGING_PHRASES.copy()


def count_uncertainty_ratio(text: str) -> float:
    """Ratio of uncertainty words to total words (lowercased)."""
    words = text.lower().split()
    if not words:
        return 0.0
    n = sum(1 for w in words if w in UNCERTAINTY_WORDS or w in MODAL_VERBS)
    return n / len(words)
