import numpy as np
import pytest

from black import _d1_d2, black_caplet, black_floorlet


def test_caplet_reference():
    F = 0.04031
    K = 0.03
    v = 4.75 + 0.25 / 3
    sigma = 0.2
    assert np.isclose(black_caplet(F, K, v, sigma), 0.012570, atol=1e-6)


def test_floorlet_reference():
    F = 0.04031
    K = 0.03
    v = 4.75 + 0.25 / 3
    sigma = 0.2
    assert np.isclose(black_floorlet(F, K, v, sigma), 0.002260, atol=1e-6)


def test_caplet_out_of_the_money():
    F = 0.01985
    K = 0.03
    v = 0.75 + 0.25 / 3
    sigma = 0.2
    assert np.isclose(black_caplet(F, K, v, sigma), 0.0000182, atol=1e-7)


def test_put_call_parity():
    F = np.array([0.01985, 0.02485, 0.02985, 0.03485, 0.03985])
    K = np.array([0.03000, 0.02500, 0.03500, 0.04000, 0.04500])
    v = np.array([0.83333, 1.08333, 1.33333, 1.58333, 1.83333])
    sigma = np.array([0.20000, 0.15000, 0.25000, 0.30000, 0.35000])
    black_results = black_caplet(F, K, v, sigma) - black_floorlet(F, K, v, sigma)
    assert np.all(np.isclose(black_results, F - K, atol=1e-12))


def test_zero_vol_limit():
    F = np.array([0.04, 0.02])
    K = np.array([0.03, 0.03])
    v = np.array([1, 1])
    sigma = np.array([1e-8, 1e-8])
    black_expected = np.array([0.01, 0])
    assert np.all(np.isclose(black_caplet(F, K, v, sigma), black_expected, atol=1e-10))


def test_increasing_in_vol():
    F = 0.04031
    K = 0.03
    v = 4.75 + 0.25 / 3
    sigma = np.array([0.1, 0.2, 0.3])
    prices = black_caplet(F, K, v, sigma)
    assert np.all(np.diff(prices) > 0)


def test_vectorized():
    F = np.array([0.01985, 0.02485, 0.02985])
    K = 0.03
    v = 4.75 + 0.25 / 3
    sigma = 0.2
    caplet_results = black_caplet(F, K, v, sigma)
    caplet_expected = np.array([black_caplet(f, K, v, sigma) for f in F])
    assert len(caplet_results) == 3
    assert np.all(np.isclose(caplet_results, caplet_expected, atol=1e-12))


def test_negative_forward_raises():
    F = -0.01
    K = 0.03
    v = 1.0
    sigma = 0.2
    with pytest.raises(ValueError, match="F must be strictly positive"):
        _d1_d2(F, K, v, sigma)


def test_zero_strike_raises():
    F = 0.03
    K = 0.0
    v = 1.0
    sigma = 0.2
    with pytest.raises(ValueError, match="K must be strictly positive"):
        _d1_d2(F, K, v, sigma)


def test_zero_time_raises():
    F = 0.03
    K = 0.03
    v = 0.0
    sigma = 0.2
    with pytest.raises(ValueError, match="v must be strictly positive"):
        _d1_d2(F, K, v, sigma)


def test_zero_vol_raises():
    F = 0.03
    K = 0.03
    v = 1.0
    sigma = 0.0
    with pytest.raises(ValueError, match="sigma must be strictly positive"):
        _d1_d2(F, K, v, sigma)
