"""Tests for backtesting."""
import pandas as pd


def test_signal_constructor():
    from backtesting.signal_constructor import construct_signal
    df = pd.DataFrame({"tft_pred_20d": [0.1, 0.2, -0.2, -0.1, 0.05]})
    sig = construct_signal(df, long_quantile=0.8, short_quantile=0.2)
    assert sig.isin([-1, 0, 1]).all()
    assert (sig == 1).sum() >= 1
    assert (sig == -1).sum() >= 1


def test_performance_metrics():
    from backtesting.performance_metrics import sharpe_ratio, max_drawdown
    import numpy as np
    ret = pd.Series([0.01, -0.005, 0.02, 0.0])
    s = sharpe_ratio(ret)
    assert isinstance(s, (int, float))
    cum = (1 + ret).cumprod()
    dd = max_drawdown(cum)
    assert dd <= 0
