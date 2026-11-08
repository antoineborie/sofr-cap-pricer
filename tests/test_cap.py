import numpy as np
import pytest

from cap import cap_price, price_from_quotes, swap_pv
from curve import TAU_QUARTER, Curve, bootstrap_swap_curve, forward_rates


@pytest.fixture
def curve():
    maturities = np.array([1, 2, 3, 4, 5])
    swap_rates = np.array([0.02, 0.0225, 0.025, 0.0275, 0.03])
    nodes, dfs = bootstrap_swap_curve(maturities, swap_rates)
    return Curve(nodes, dfs)


def test_cap_total(curve):
    N = 10e6
    K = 0.03
    sigma = 0.2
    contributions_sum, contributions = cap_price(N, K, curve, sigma, kind="cap")
    assert np.isclose(contributions_sum, 223252, atol=1)


def test_floor_total(curve):
    N = 10e6
    K = 0.03
    sigma = 0.2
    contributions_sum, contributions = cap_price(N, K, curve, sigma, kind="floor")
    assert np.isclose(contributions_sum, 239168, atol=1)


def test_swap_pv(curve):
    N = 10e6
    K = 0.03
    swap_pv_value = swap_pv(N, K, curve)
    assert np.isclose(swap_pv_value, -15916, atol=1)


def test_last_caplet(curve):
    N = 10e6
    K = 0.03
    sigma = 0.2
    contributions_sum, contributions = cap_price(N, K, curve, sigma, kind="cap")
    assert np.isclose(contributions[-1], 27386, atol=1)


def test_contributions_consistency(curve):
    N = 10e6
    K = 0.03
    sigma = 0.2
    contributions_sum, contributions = cap_price(N, K, curve, sigma, kind="cap")
    assert len(contributions) == 20
    assert np.isclose(contributions.sum(), contributions_sum, atol=1e-8)


def test_contributions_positive(curve):
    N = 10e6
    K = 0.03
    sigma = 0.2
    contributions_sum, contributions = cap_price(N, K, curve, sigma, kind="cap")
    assert np.all(contributions > 0)


def test_put_call_parity(curve):
    N = 10e6
    sigmas = np.array([0.1, 0.2, 0.5])
    strikes = np.array([0.02, 0.03, 0.04])
    for sigma in sigmas:
        for K in strikes:
            cap = cap_price(N, K, curve, sigma, kind="cap")[0]
            floor = cap_price(N, K, curve, sigma, kind="floor")[0]
            swap = swap_pv(N, K, curve)
            assert np.isclose(cap - floor, swap, atol=1e-6)


def test_atm_strike(curve):
    N = 10e6
    sigma = 0.2
    grid = np.arange(0, 21) * 0.25
    dfs = curve.df(grid[1:])
    K_atm = (1 - curve.df(5.0)) / (TAU_QUARTER * np.sum(dfs))
    cap = cap_price(N, K_atm, curve, sigma, kind="cap")[0]
    floor = cap_price(N, K_atm, curve, sigma, kind="floor")[0]
    assert np.isclose(K_atm, 0.029664, atol=1e-5)
    assert np.isclose(swap_pv(N, K_atm, curve), 0, atol=1e-6)
    assert np.isclose(cap, floor, atol=1e-6)


def test_zero_vol(curve):
    N = 10e6
    K = 0.03
    sigma = 1e-8
    grid = np.arange(0, 21) * 0.25
    F = forward_rates(curve, grid)
    dfs = curve.df(grid[1:])
    expected = N * TAU_QUARTER * np.sum(dfs * np.maximum(F - K, 0))
    contributions_sum, contributions = cap_price(N, K, curve, sigma, kind="cap")
    assert np.isclose(contributions_sum, expected, atol=1e-6)


def test_increasing_in_vol(curve):
    N = 10e6
    K = 0.03
    sigmas = np.array([0.1, 0.2, 0.3])
    prices = np.array([cap_price(N, K, curve, sigma)[0] for sigma in sigmas])
    assert np.all(np.diff(prices) > 0)


def test_invalid_kind(curve):
    N = 10e6
    K = 0.03
    sigma = 0.2
    with pytest.raises(ValueError, match="kind must be cap or floor"):
        cap_price(N, K, curve, sigma, kind="swap")


def test_price_from_quotes(curve):
    quotes = np.array([0.02, 0.0225, 0.025, 0.0275, 0.03])
    sigma = 0.2
    cap = cap_price(10e6, 0.03, curve, sigma, kind="cap")[0]
    floor = cap_price(10e6, 0.03, curve, sigma, kind="floor")[0]
    assert np.isclose(price_from_quotes(quotes, sigma), cap, atol=1e-8)
    assert np.isclose(price_from_quotes(quotes, sigma, kind="floor"), floor, atol=1e-8)
    assert np.isclose(price_from_quotes(list(quotes), sigma), cap, atol=1e-8)
