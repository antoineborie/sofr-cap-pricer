import numpy as np

TAU_ANNUAL = 365 / 360
TAU_QUARTER = 91.25 / 360


def interpolate_swap_rates(known_maturities, known_rates, maturities):
    """This function :
    - converts the inputs into arrays of float
    - checks that the known maturities are striclty increasing
    - checks that min and max values are in the known range
    - return the interpolated rates"""

    known_maturities = np.asarray(known_maturities, dtype=float)
    known_rates = np.asarray(known_rates, dtype=float)
    maturities = np.asarray(maturities, dtype=float)

    if not np.all(np.diff(known_maturities) > 0):
        raise ValueError("known_maturities must be strictly increasing")

    if maturities.max() > known_maturities.max() or maturities.min() < known_maturities.min():
        raise ValueError(
            f"maturities must lie within [{known_maturities.min()}, {known_maturities.max()}]"
        )

    return np.interp(maturities, known_maturities, known_rates)


def bootstrap_swap_curve(maturities, swap_rates, tau=TAU_ANNUAL):
    """This function :
    - converts inputs into arrays of floats
    - checks if maturities and swap_rates have the same lenght
    - checks maturities are exactly 1, 2, ...
    - bootstraps the df at each maturity
    - return two arrays: one with maturities, the other one with calculated dfs"""

    maturities = np.asarray(maturities, dtype=float)
    swap_rates = np.asarray(swap_rates, dtype=float)
    if len(maturities) != len(swap_rates):
        raise ValueError("maturities has a different size from swap rates")

    n = len(maturities)
    if not np.allclose(maturities, np.arange(1, n + 1)):
        raise ValueError("maturities are not 1, 2, ..., N")
    dfs = np.zeros(n)
    annuity = 0.0

    for i, s in enumerate(swap_rates):
        df = (1 - s * annuity) / (1 + s * tau)
        dfs[i] = df
        annuity += tau * df

    return np.concatenate([[0.0], maturities]), np.concatenate([[1.0], dfs])


class Curve:
    """function __init__:
    -create an object for nodes and log_dfs
    function df:
    -discount factor at time t, by linear interpolation of log DF
    function zero:
    -continuously compounded zero rate at time
    """

    def __init__(self, nodes, dfs):
        nodes = np.asarray(nodes, dtype=float)
        dfs = np.asarray(dfs, dtype=float)
        if len(nodes) != len(dfs):
            raise ValueError("nodes and dfs must have the same length")
        if not np.all(np.diff(nodes) > 0):
            raise ValueError("nodes must be strictly increasing")
        if not np.isclose(nodes[0], 0.0):
            raise ValueError("nodes[0] must equal 0")
        if not np.isclose(dfs[0], 1.0, atol=1e-12):
            raise ValueError("dfs[0] must equal 1")
        if not np.all(dfs > 0):
            raise ValueError("all discount factors must be positive")
        self.nodes = nodes
        self.log_dfs = np.log(dfs)

    def df(self, t):
        """this function :
        - converts inputs into array of floats
        - checks t is positive and it is in the nodes's range
        -  linear interpolation of DF at date t
        """
        t = np.asarray(t, dtype=float)
        if np.any(t < 0):
            raise ValueError("t must be non-negative")
        if np.any(t > self.nodes[-1]):
            raise ValueError(f"t must not exceed the last node ({self.nodes[-1]})")
        return np.exp(np.interp(t, self.nodes, self.log_dfs))

    def zero(self, t):
        """this functiuon:
        - converts inputs into array of floats
        - checks if t is positive
        - calculates zero rate from df with date t
        """
        t = np.asarray(t, dtype=float)
        if np.any(t <= 0):
            raise ValueError("t must be strictly positive")
        return -np.log(self.df(t)) / t


def forward_rates(curve, t_grid):
    """this function :
    - converts t_grid into array of floats
    - checks grid has the min size and increases
    - interpolates dfs relative to t_grid
    - computes act/360 for each period
    - computes forward rates
    """
    t_grid = np.asarray(t_grid, dtype=float)
    if len(t_grid) < 2:
        raise ValueError("t_grid is too short 2<")
    if not np.all(np.diff(t_grid) > 0):
        raise ValueError("t_grid must be strictly increasing")
    dfs = curve.df(t_grid)
    taus = np.diff(t_grid) * TAU_ANNUAL
    forward_rates_values = (dfs[:-1] / dfs[1:] - 1) / taus
    return forward_rates_values


def par_swap_rate(curve, maturity, tau=TAU_ANNUAL):
    """this function:
    - checks if maturity at least 1
    - checks if maturity is int
    - creates payment dates from issuance to maturity
    - calculates the par_swap_rate
    """
    if maturity < 1:
        raise ValueError("maturity must be at least 1")
    if not np.isclose(maturity, round(maturity)):
        raise ValueError("maturity must be an int")
    payment_date = np.arange(1, int(maturity) + 1)
    df_payment_date = curve.df(payment_date)
    annuity = tau * np.sum(df_payment_date)
    return (1 - df_payment_date[-1]) / annuity
