"""
Per-call explanation examples (stub for waterfall / plain-English summary).
"""
from __future__ import annotations

from pathlib import Path


def explain_call(call_id: str, model_type: str = "xgboost") -> dict:
    """Return a short explanation dict for one call (stub)."""
    return {"call_id": call_id, "top_driver": "sentiment_trajectory_slope", "effect": 0.01}


if __name__ == "__main__":
    print(explain_call("AAPL_2024-02-01"))
