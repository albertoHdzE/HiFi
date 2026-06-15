"""Tests for DriftMonitor statistical tests (P13-E5, DJ-077)."""

import numpy as np
import pandas as pd
import pytest

from hifi.collective.drift import DriftMonitor, DriftResult


@pytest.fixture
def monitor():
    return DriftMonitor()


# ---------------------------------------------------------------------------
# DriftResult schema
# ---------------------------------------------------------------------------

def test_drift_result_fields():
    r = DriftResult(drift_type="data", statistic=0.03, p_value=0.03, alert=True, threshold=0.05)
    assert r.drift_type == "data"
    assert r.alert is True


# ---------------------------------------------------------------------------
# E5-T2: check_data_drift (KS)
# ---------------------------------------------------------------------------

def test_data_drift_identical_distributions_no_alert(monitor):
    rng = np.random.default_rng(42)
    data = pd.DataFrame({"vol": rng.normal(0.2, 0.05, 100)})
    result = monitor.check_data_drift(data, data.copy())
    assert result.drift_type == "data"
    assert not result.alert


def test_data_drift_different_distributions_alert(monitor):
    rng = np.random.default_rng(0)
    baseline = pd.DataFrame({"vol": rng.normal(0.2, 0.05, 200)})
    recent = pd.DataFrame({"vol": rng.normal(0.8, 0.05, 50)})
    result = monitor.check_data_drift(recent, baseline)
    assert result.alert
    assert "vol" in result.feature_alerts


def test_data_drift_no_shared_columns_no_alert(monitor):
    recent = pd.DataFrame({"a": [1.0, 2.0]})
    baseline = pd.DataFrame({"b": [3.0, 4.0]})
    result = monitor.check_data_drift(recent, baseline)
    assert not result.alert


def test_data_drift_multiple_features_vol_alerts(monitor):
    rng = np.random.default_rng(7)
    n = 500
    pe_base = rng.normal(20.0, 2.0, n)
    baseline = pd.DataFrame({"vol": rng.normal(0.2, 0.05, n), "pe": pe_base})
    # vol is clearly shifted; pe is literally the same data subset
    recent = pd.DataFrame({"vol": rng.normal(0.9, 0.05, 50), "pe": pe_base[:50]})
    result = monitor.check_data_drift(recent, baseline)
    assert result.alert
    assert "vol" in result.feature_alerts


def test_data_drift_p_value_in_result(monitor):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(0, 1, 100)})
    result = monitor.check_data_drift(df, df.copy())
    assert result.p_value is not None
    assert 0.0 <= result.p_value <= 1.0


# ---------------------------------------------------------------------------
# E5-T3: check_agent_drift (chi-squared)
# ---------------------------------------------------------------------------

_BASELINE = {"Buy": 0.45, "Hold": 0.38, "Sell": 0.17}


def test_agent_drift_matching_distribution_no_alert(monitor):
    # Generate decisions that match the baseline proportions
    decisions = ["Buy"] * 45 + ["Hold"] * 38 + ["Sell"] * 17
    result = monitor.check_agent_drift(decisions, _BASELINE)
    assert result.drift_type == "agent"
    assert not result.alert


def test_agent_drift_all_buy_alerts(monitor):
    decisions = ["Buy"] * 50
    result = monitor.check_agent_drift(decisions, _BASELINE)
    assert result.alert


def test_agent_drift_empty_decisions_no_alert(monitor):
    result = monitor.check_agent_drift([], _BASELINE)
    assert not result.alert
    assert result.p_value == pytest.approx(1.0)


def test_agent_drift_statistic_positive(monitor):
    decisions = ["Buy"] * 45 + ["Hold"] * 38 + ["Sell"] * 17
    result = monitor.check_agent_drift(decisions, _BASELINE)
    assert result.statistic >= 0.0


def test_agent_drift_zero_baseline_no_alert(monitor):
    result = monitor.check_agent_drift(["Buy", "Hold"], {"Buy": 0.0, "Hold": 0.0, "Sell": 0.0})
    assert not result.alert


# ---------------------------------------------------------------------------
# E5-T4: check_collective_drift (CUSUM)
# ---------------------------------------------------------------------------

def test_collective_drift_stable_series_no_alert(monitor):
    herding = [0.82] * 20
    result = monitor.check_collective_drift(herding, baseline_mean=0.82, baseline_std=0.05)
    assert result.drift_type == "collective"
    assert not result.alert


def test_collective_drift_elevated_series_alerts(monitor):
    herding = [1.0] * 30
    result = monitor.check_collective_drift(herding, baseline_mean=0.82, baseline_std=0.05)
    assert result.alert


def test_collective_drift_p_value_none(monitor):
    result = monitor.check_collective_drift([0.8], baseline_mean=0.8, baseline_std=0.05)
    assert result.p_value is None


def test_collective_drift_empty_series_no_alert(monitor):
    result = monitor.check_collective_drift([], baseline_mean=0.82, baseline_std=0.05)
    assert not result.alert


def test_collective_drift_zero_std_no_alert(monitor):
    result = monitor.check_collective_drift([0.9, 0.9], baseline_mean=0.82, baseline_std=0.0)
    assert not result.alert


def test_collective_drift_statistic_non_negative(monitor):
    result = monitor.check_collective_drift([0.7, 0.8, 0.9], baseline_mean=0.82, baseline_std=0.05)
    assert result.statistic >= 0.0
