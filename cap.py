import numpy as np

from black import black_caplet, black_floorlet
from curve import TAU_ANNUAL, TAU_QUARTER, Curve, bootstrap_swap_curve, forward_rates


def cap_price(N, K, curve, sigma, kind="cap"):
    """This function:
    - selects the formula to call (cap or floor)
    - calculates the total cap/floor price by summing the discounted values of all caplets/floorlets
    """
    if kind == "cap":
        formula = black_caplet
    elif kind == "floor":
        formula = black_floorlet
    else:
        raise ValueError("kind must be cap or floor")
    grid = np.arange(0, 21) * 0.25
    F = forward_rates(curve, grid)
    dfs = curve.df(grid[1:])
    v = grid[:-1] + (0.25 / 3)
    undiscounted = formula(F, K, v, sigma)
    contributions = N * TAU_QUARTER * dfs * undiscounted
    return contributions.sum(), contributions


def swap_pv(N, K, curve):
    """This function:
    - calculates the present value of a quarterly payer swap
    - discounts and sums the net cash flows
    """
    grid = np.arange(0, 21) * 0.25
    F = forward_rates(curve, grid)
    dfs = curve.df(grid[1:])
    PV = N * TAU_QUARTER * np.sum(dfs * (F - K))
    return PV


def price_from_quotes(swap_rates, sigma, N=10e6, K=0.03, kind="cap"):
    """This function:
    - bootstrap a discount curve from market swap rate quotes
    - construct a yield curve from the resulting discount factor
    - price cap or floor
    - returns the total instrument price
    """
    swap_rates = np.asarray(swap_rates, dtype=float)
    maturities = np.arange(1, len(swap_rates) + 1)
    nodes, dfs = bootstrap_swap_curve(maturities, swap_rates, TAU_ANNUAL)
    curve = Curve(nodes, dfs)
    price = cap_price(N, K, curve, sigma, kind=kind)[0]
    return price
