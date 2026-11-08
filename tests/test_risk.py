import numpy as np
import pytest

from risk import bucket_deltas, dv01, steepener, vega


@pytest.fixture
def quotes():
    return np.array([0.02, 0.0225, 0.025, 0.0275, 0.03])


def test_dv01(quotes):
    sigma = 0.2
    assert np.isclose(dv01(quotes, sigma), 2066, atol=1)


def test_bucket_deltas(quotes):
    sigma = 0.2
    deltas = bucket_deltas(quotes, sigma)
    expected = np.array([-248, -605, -501, -362, 3781])
    assert np.all(np.isclose(deltas, expected, atol=1))


def test_vega(quotes):
    sigma = 0.2
    assert np.isclose(vega(quotes, sigma), 6678, atol=1)


def test_buckets_sum_to_dv01(quotes):
    sigma = 0.2
    deltas = bucket_deltas(quotes, sigma)
    assert np.isclose(np.sum(deltas), dv01(quotes, sigma), atol=2)


def test_steepener_matches_deltas(quotes):
    sigma = 0.2
    deltas = bucket_deltas(quotes, sigma)
    shock_in_bp = np.array([-1, -1, 1, 1, 1])
    expected = np.sum(deltas * shock_in_bp)
    up = steepener(quotes, sigma, 1e-4)
    down = steepener(quotes, sigma, -1e-4)
    assert np.isclose((up - down) / 2, expected, atol=2)


def test_steepener_convexity(quotes):
    sigma = 0.2
    up = steepener(quotes, sigma, 1e-4)
    down = steepener(quotes, sigma, -1e-4)
    assert up + down > 0


def test_quotes_unchanged(quotes):
    sigma = 0.2
    original = quotes.copy()
    bucket_deltas(quotes, sigma)
    assert np.array_equal(quotes, original)


def test_signs(quotes):
    sigma = 0.2
    deltas = bucket_deltas(quotes, sigma)
    assert dv01(quotes, sigma) > 0
    assert vega(quotes, sigma) > 0
    assert np.argmax(deltas) == 4
