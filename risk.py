import numpy as np

from cap import price_from_quotes


def dv01(swap_rates, sigma, h=1e-4):
    """This function:
    - Adds 1 bp on all rate in the same time, build the curve and price the cap
    - Subtracts 1 bp on all rate in the same time, build the curve and price the floor
    """
    swap_rates = np.asarray(swap_rates, dtype=float)
    up = price_from_quotes(swap_rates + h, sigma)
    down = price_from_quotes(swap_rates - h, sigma)
    return (up - down) / 2


def bucket_deltas(swap_rates, sigma, h=1e-4):
    """This function:
    - Copies the rates and adds 1 bp only th the i-th one
    - it copies them again and subtracts 1 bp from the i-th one
    - Reprice the two scenarios
    """
    swap_rates = np.asarray(swap_rates, dtype=float)
    deltas = np.zeros(len(swap_rates))
    for i in range(len(swap_rates)):
        up_rates = swap_rates.copy()
        up_rates[i] = up_rates[i] + h
        down_rates = swap_rates.copy()
        down_rates[i] = down_rates[i] - h
        deltas[i] = (price_from_quotes(up_rates, sigma) - price_from_quotes(down_rates, sigma)) / 2
    return deltas


def vega(swap_rates, sigma, h=0.01):
    """This function:
    - reprice with higher sigma
    - reprice with lower sigma
    - return an amount per point of vol
    """
    up = price_from_quotes(swap_rates, sigma + h)
    down = price_from_quotes(swap_rates, sigma - h)
    return (up - down) / 2


def steepener(swap_rates, sigma, shock=1e-4):
    swap_rates = np.asarray(swap_rates, dtype=float)
    move = np.array([-shock, -shock, shock, shock, shock])
    return price_from_quotes(swap_rates + move, sigma) - price_from_quotes(swap_rates, sigma)
