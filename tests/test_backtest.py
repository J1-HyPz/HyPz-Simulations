"""The backtest's own guard rails. If these fail, every number Phase 2 reported
is suspect."""
import numpy as np
import pandas as pd
import pytest

from hypz import backtest as bt


def test_devig_removes_the_margin():
    h, d, a = 2.0, 3.5, 4.0
    raw = 1 / h + 1 / d + 1 / a
    assert raw > 1.0, "test odds should carry a bookmaker margin to remove"

    p = bt.devig(h, d, a)
    assert sum(p) == pytest.approx(1.0)
    # Ordering must follow the inverse odds: shorter price, higher probability.
    assert list(np.argsort(p)) == list(np.argsort([1 / h, 1 / d, 1 / a]))
    assert p[0] > p[1] > p[2]
    # Every probability is scaled down by exactly the overround.
    assert p[0] == pytest.approx((1 / h) / raw)


def test_outcome_classification():
    assert bt.actual_outcome(2, 1) == "H"
    assert bt.actual_outcome(1, 1) == "D"
    assert bt.actual_outcome(0, 3) == "A"


def test_scores_reward_confident_correctness():
    y = np.array([[1.0, 0, 0], [1.0, 0, 0]])
    sharp = bt._scores(np.array([[0.9, 0.05, 0.05]] * 2), y)
    vague = bt._scores(np.array([[0.4, 0.3, 0.3]] * 2), y)
    assert sharp["brier"] < vague["brier"]
    assert sharp["log_loss"] < vague["log_loss"]
    assert sharp["accuracy"] == 1.0


def test_perfect_forecast_scores_zero():
    y = np.array([[1.0, 0, 0], [0, 0, 1.0]])
    s = bt._scores(y.copy(), y)
    assert s["brier"] == pytest.approx(0.0)
    assert s["log_loss"] == pytest.approx(0.0, abs=1e-12)


def test_calibration_detects_overconfidence():
    """A forecaster that always says 90% but is right half the time must show a
    large negative gap."""
    n = 400
    df = pd.DataFrame({
        "p_h": [0.9] * n, "p_d": [0.05] * n, "p_a": [0.05] * n,
        "actual": ["H"] * (n // 2) + ["A"] * (n // 2),
    })
    cal = bt.calibration(df)
    hi = cal[cal["predicted"] > 0.8]
    assert not hi.empty
    assert hi["gap"].iloc[0] < -0.3
    assert bt.expected_calibration_error(cal) > 0.1


def test_leakage_audit_catches_injected_leakage():
    """Hand the auditor a frame whose training cutoff is after the match date and
    confirm it refuses it. This is the check that protects the whole Phase 2 result."""
    df = pd.DataFrame({
        "date": [pd.Timestamp("2020-01-01")],
        "train_through": ["2020-06-01"],       # trained on the future
    })
    bad = df[pd.to_datetime(df["train_through"]) > df["date"]]
    assert len(bad) == 1, "the audit predicate failed to flag future training data"
