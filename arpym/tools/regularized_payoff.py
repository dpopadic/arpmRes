# -*- coding: utf-8 -*-"
from numpy import pi, exp, sqrt
from scipy.special import erf


def regularized_payoff(x, k_strk, h, method):
    """For details, see here:

    Parameters
    ----------
        x : array, shape (j_,)
        k : scalar
        method: string
        h: scalar

    Returns
    -------
        v_h : array, shape (j_,)

    """

    # Step 1: Compute the payoff

    if method == "call":
        v_h = h / sqrt(2 * pi) * exp(-(k_strk - x)**2 / (2 * h**2)) \
            + (x - k_strk) / 2 * (1 + erf((x - k_strk) / sqrt(2 * h**2)))

    elif method == "put":
        v_h = h / sqrt(2 * pi) * exp(-(x - k_strk)**2 / (2 * h**2)) \
            - (x - k_strk)/2 * (1 - erf((x - k_strk)/sqrt(2 * h**2)))

    return v_h
