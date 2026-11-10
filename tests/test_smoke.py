import numpy as np
import pytest

from comergy import allocation as A
from comergy.config import BatteryParams
from comergy.data_loader import load_dataset
from comergy.optimizer import solve_battery
from comergy.scenarios import baseline_cost


@pytest.fixture(scope="module")
def data():
    return load_dataset()


def test_baseline_matches_report(data):
    assert abs(baseline_cost(data)["M1"] - 1720) < 1
    assert abs(baseline_cost(data).sum() - 22613) < 2


def test_lp_feasible_and_balanced(data):
    net = (data.demand["M3"] - data.pv["M3"]).values[:48]
    s = solve_battery(net, data.price[:48], data.fit[:48], 13.98, BatteryParams())
    assert np.allclose(s.imp - s.exp - s.ch + s.dis, net, atol=1e-6)
    assert s.soc.min() >= 0.1 * 13.98 - 1e-6


def test_shapley_efficiency_small_game():
    C = np.array([0, 10, 10, 15.0])        # two players: v({1,2}) = 20-15 = 5
    phi, v, diag = A.shapley_savings(C, 2)
    assert abs(phi.sum() - 5) < 1e-9 and np.allclose(phi, [2.5, 2.5])


def test_no_worse_off():
    import pandas as pd
    ref = pd.Series([100.0, 100.0, 100.0])
    bills = pd.Series([50.0, 130.0, 60.0])
    out = A.enforce_no_worse_off(bills, ref)
    assert (out <= ref + 1e-9).all() and abs(out.sum() - bills.sum()) < 1e-9


def test_neural_forecaster_runs(data):
    """The MLP forecaster returns finite, non-negative out-of-fold forecasts for GHI (fast check)."""
    from comergy.forecasting import _oof_predict, ghi_features
    p = _oof_predict(ghi_features(data), data.ghi, 3, 0, kind="mlp")
    assert np.isfinite(p).all() and (p >= 0).all() and p.shape == data.ghi.shape
