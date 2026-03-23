"""Tests for causal module."""
import pandas as pd


def test_event_study_class():
    from causal.event_study import EventStudy
    es = EventStudy(estimation_window=60, event_windows=[(0, 1), (0, 5)])
    assert es.estimation_window == 60
    assert len(es.event_windows) == 2


def test_bootstrap_car():
    from causal.significance_tests import bootstrap_car
    import numpy as np
    cars = np.array([0.01, -0.02, 0.015, 0.0, -0.01])
    res = bootstrap_car(cars, n_bootstrap=100)
    assert "mean" in res
    assert res["ci_low"] <= res["mean"] <= res["ci_high"]
