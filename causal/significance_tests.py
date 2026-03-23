"""
Bootstrap and permutation tests for event study and DiD.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def bootstrap_car(cars: np.ndarray, n_bootstrap: int = 5000, ci: float = 0.95) -> dict[str, float]:
    """Bootstrap CI for mean CAR."""
    n = len(cars)
    if n == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    rng = np.random.default_rng(42)
    boot = rng.choice(cars, size=(n_bootstrap, n), replace=True).mean(axis=1)
    alpha = 1 - ci
    return {
        "mean": float(np.mean(cars)),
        "ci_low": float(np.quantile(boot, alpha / 2)),
        "ci_high": float(np.quantile(boot, 1 - alpha / 2)),
    }


def permutation_test_treatment(treated: np.ndarray, control: np.ndarray, n_perm: int = 5000) -> float:
    """Two-sample permutation p-value (treatment vs control CAR)."""
    combined = np.concatenate([treated, control])
    n_t = len(treated)
    obs_diff = np.mean(treated) - np.mean(control)
    rng = np.random.default_rng(42)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(combined)
        diff = np.mean(perm[:n_t]) - np.mean(perm[n_t:])
        if diff <= obs_diff:
            count += 1
    return (count + 1) / (n_perm + 1)
