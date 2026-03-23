"""
Realistic cost modelling: bps per side, slippage.
"""
from __future__ import annotations


def cost_per_side(bps: float = 5.0, slippage_bps: float = 2.0) -> float:
    return (bps + slippage_bps) / 1e4


def round_trip_cost(bps: float = 5.0, slippage_bps: float = 2.0) -> float:
    return 2 * cost_per_side(bps, slippage_bps)
