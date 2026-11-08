import numpy as np
from scipy.stats import norm


def _d1_d2(F, K, v, sigma):
    """This function:
    - converts inputs into arrays of floats
    - calculates d1, representing the standardized log-moneyness adjusted for volatility
    - calculates d2, representing the standardized log-moneyness adjusted for volatility and time to maturity
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    v = np.asarray(v, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    if np.any(F <= 0):
        raise ValueError("F must be strictly positive")
    if np.any(K <= 0):
        raise ValueError("K must be strictly positive")
    if np.any(v <= 0):
        raise ValueError("v must be strictly positive")
    if np.any(sigma <= 0):
        raise ValueError("sigma must be strictly positive")

    sv = sigma * np.sqrt(v)
    d1 = (np.log(F / K) + 1 / 2 * sv**2) / sv
    d2 = d1 - sv
    return d1, d2


def black_caplet(F, K, v, sigma):
    """This function:
    - calculates the undiscounted Black-76 caplet value return following a lognormal distribution
    """
    d1, d2 = _d1_d2(F, K, v, sigma)
    black_cap_value = F * norm.cdf(d1) - K * norm.cdf(d2)
    return black_cap_value


def black_floorlet(F, K, v, sigma):
    """This function:
    - calculates the undiscounted Black-76 floorlet value return following a lognormal distribution
    """
    d1, d2 = _d1_d2(F, K, v, sigma)
    black_floor_value = K * norm.cdf(-d2) - F * norm.cdf(-d1)
    return black_floor_value
