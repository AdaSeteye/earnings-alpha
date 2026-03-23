"""
Model versioning and load/save helpers.
"""
from __future__ import annotations

from pathlib import Path


def get_model_dir(name: str) -> Path:
    return Path("models") / name


def register_model(name: str, version: str, path: str) -> None:
    try:
        import mlflow
        mlflow.set_tracking_uri(__import__("os").environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.register_model(path, name)
    except Exception:
        pass
