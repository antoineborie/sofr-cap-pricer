import numpy as np
import pytest

from curve import (
    TAU_ANNUAL,
    TAU_QUARTER,
    Curve,
    bootstrap_swap_curve,
    forward_rates,
    interpolate_swap_rates,
    par_swap_rate,
)


def make_project_curve():
    nodes, dfs = bootstrap_swap_curve(PROJECT_MAT, PROJECT_RATES)
    return Curve(nodes, dfs)


KNOWN_MAT = [1, 3, 5]
KNOWN_RATES = [0.02, 0.025, 0.03]
PROJECT_MAT = [1, 2, 3, 4, 5]
PROJECT_RATES = [0.02, 0.0225, 0.025, 0.0275, 0.03]
QUARTERLY_GRID = np.arange(0, 21) * 0.25


def test_interpolate_project_rates():
    maturities = [1, 2, 3, 4, 5]
    expected_result = [0.02, 0.0225, 0.025, 0.0275, 0.03]
    real_result = interpolate_swap_rates(KNOWN_MAT, KNOWN_RATES, maturities)
    assert np.allclose(real_result, expected_result, atol=1e-12), (
        f"interpolate expected result {expected_result} is different from the real one {real_result}"
    )


def test_interpolate_subset():
    maturities = [2, 3]
    expected_result = [0.0225, 0.025]
    real_result = interpolate_swap_rates(KNOWN_MAT, KNOWN_RATES, maturities)
    assert np.allclose(real_result, expected_result, atol=1e-12), (
        f"interpolate expected result {expected_result} is different from the real one {real_result}"
    )


def test_interpolate_single_maturity():
    maturities = 2
    expected_result = 0.0225
    real_result = interpolate_swap_rates(KNOWN_MAT, KNOWN_RATES, maturities)
    assert np.allclose(real_result, expected_result, atol=1e-12), (
        f"interpolate expected result {expected_result} is different from the real one {real_result}"
    )


def test_interpolate_around_range():
    maturities = [7]
    with pytest.raises(ValueError, match="maturities must lie within"):
        interpolate_swap_rates(KNOWN_MAT, KNOWN_RATES, maturities)
    maturities = [0.5]
    with pytest.raises(ValueError, match="maturities must lie within"):
        interpolate_swap_rates(KNOWN_MAT, KNOWN_RATES, maturities)


def test_interpolate_non_increasing():
    known_maturities = [1, 5, 3]
    maturities = [2, 3]
    with pytest.raises(ValueError, match="strictly increasing"):
        (
            interpolate_swap_rates(known_maturities, KNOWN_RATES, maturities),
            f"known_maturities {known_maturities} are not strictly increasing",
        )


def test_bootstrap_nodes():
    nodes, dfs = bootstrap_swap_curve(PROJECT_MAT, PROJECT_RATES)
    expected_nodes = [0, 1, 2, 3, 4, 5]
    assert np.allclose(nodes, expected_nodes), (
        f"nodes from bootstrap are different from {expected_nodes}"
    )


def test_bootstrap_dfs():
    nodes, dfs = bootstrap_swap_curve(PROJECT_MAT, PROJECT_RATES)
    expected_dfs = [1, 0.980125, 0.955836, 0.927421, 0.895203, 0.859532]
    assert np.allclose(dfs, expected_dfs), f"dfs from boostrap are different from {expected_dfs}"


def test_boostrap_repricing():
    nodes, dfs = bootstrap_swap_curve(PROJECT_MAT, PROJECT_RATES)
    annuity = np.cumsum(TAU_ANNUAL * dfs[1:])
    n = len(annuity)
    computed_swap_array = np.zeros(n)
    for i in range(n):
        computed_swap_array[i] = (1 - dfs[i + 1]) / annuity[i]
    assert np.allclose(PROJECT_RATES, computed_swap_array, atol=1e-12)


def test_bootstrap_flat_curve():
    flat_rates = np.full(5, 0.03)
    nodes, dfs = bootstrap_swap_curve(PROJECT_MAT, flat_rates)
    zero_rates = -np.log(dfs[1:]) / nodes[1:]
    expected_value = np.log(1 + 0.03 * TAU_ANNUAL)
    assert np.allclose(zero_rates, expected_value, atol=1e-12)


def test_bootstrap_missing_maturity():
    with pytest.raises(ValueError, match="maturities are not 1, 2, ..., N"):
        bootstrap_swap_curve(KNOWN_MAT, KNOWN_RATES)


def test_bootstrap_lenght_mismatch():
    with pytest.raises(ValueError, match="maturities has a different size from swap rates"):
        bootstrap_swap_curve([1, 2, 3], [0.02, 0.0225])


def test_curve_df_at_nodes():
    nodes, dfs = bootstrap_swap_curve(PROJECT_MAT, PROJECT_RATES)
    curve = Curve(nodes, dfs)
    assert np.allclose(curve.df(nodes), dfs, atol=1e-12)


def test_curve_df_interpolated():
    curve = make_project_curve()
    assert np.isclose(curve.df(0.5), 0.990013, atol=1e-6)


def test_curve_zero_at_nodes():
    curve = make_project_curve()
    expected = [0.020075, 0.022585, 0.025116, 0.027676, 0.030273]
    assert np.allclose(curve.zero([1, 2, 3, 4, 5]), expected, atol=1e-6)


def test_curve_df_above_range():
    curve = make_project_curve()
    with pytest.raises(ValueError, match="must not exceed"):
        curve.df(5.5)


def test_curve_df_negative():
    curve = make_project_curve()
    with pytest.raises(ValueError, match="non-negative"):
        curve.df(-0.1)


def test_curve_zero_at_zero():
    curve = make_project_curve()
    with pytest.raises(ValueError, match="strictly positive"):
        curve.zero(0)


def test_curve_invalid_first_df():
    with pytest.raises(ValueError, match="must equal 1"):
        Curve([0, 1], [0.99, 0.98])


def test_forward_count():
    curve = make_project_curve()
    assert len(forward_rates(curve, QUARTERLY_GRID)) == 20


def test_forward_staircase():
    curve = make_project_curve()
    F = forward_rates(curve, QUARTERLY_GRID)
    for k in range(5):
        assert np.allclose(F[4 * k : 4 * k + 4], F[4 * k], atol=1e-12)


def test_forward_levels():
    curve = make_project_curve()
    F = forward_rates(curve, QUARTERLY_GRID)
    expected = [0.01985, 0.02483, 0.02988, 0.03503, 0.04031]
    assert np.allclose(F[::4], expected, atol=1e-5)


def test_forward_recomposition():
    curve = make_project_curve()
    nodes, dfs = bootstrap_swap_curve(PROJECT_MAT, PROJECT_RATES)
    F = forward_rates(curve, QUARTERLY_GRID)
    for k in range(5):
        growth = np.prod(1 + F[4 * k : 4 * k + 4] * TAU_QUARTER)
        assert np.isclose(growth, dfs[k] / dfs[k + 1], atol=1e-12)


def test_forward_non_increasing():
    curve = make_project_curve()
    with pytest.raises(ValueError, match="strictly increasing"):
        forward_rates(curve, [0, 0.5, 0.25])


def test_par_swap_repricing():
    curve = make_project_curve()
    for n in range(1, 6):
        assert np.isclose(par_swap_rate(curve, n), PROJECT_RATES[n - 1], atol=1e-12)


def test_par_swap_non_integer():
    curve = make_project_curve()
    with pytest.raises(ValueError, match="must be an int"):
        par_swap_rate(curve, 2.5)


def test_par_swap_beyond_curve():
    curve = make_project_curve()
    with pytest.raises(ValueError, match="must not exceed"):
        par_swap_rate(curve, 6)
