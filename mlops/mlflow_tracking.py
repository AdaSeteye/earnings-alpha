"""
Experiment logging with MLflow.
"""
from __future__ import annotations

import os
from pathlib import Path


def get_tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


def log_run(name: str, params: dict, metrics: dict, artifact_path: str | None = None) -> None:
    try:
        import mlflow
        mlflow.set_tracking_uri(get_tracking_uri())
        with mlflow.start_run(run_name=name):
            for k, v in params.items():
                mlflow.log_param(k, v)
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            if artifact_path and Path(artifact_path).exists():
                mlflow.log_artifact(artifact_path)
    except Exception as e:
        print("MLflow log failed:", e)
