"""
Sharpe, max drawdown, turnover, IC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, annualise: float = 252) -> float:
    if returns.std() == 0 or len(returns) < 2:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(annualise))


def max_drawdown(cumulative_returns: pd.Series) -> float:
    peak = cumulative_returns.expanding().max()
    dd = (cumulative_returns - peak) / (peak + 1e-12)
    return float(dd.min())


def information_coefficient(signal: pd.Series, forward_returns: pd.Series) -> float:
    return float(signal.corr(forward_returns)) if len(signal) > 1 else 0.0


def annualised_turnover(weights: pd.DataFrame) -> float:
    diff = weights.diff().abs()
    return float(diff.sum(axis=1).mean() * 252) if not diff.empty else 0.0
