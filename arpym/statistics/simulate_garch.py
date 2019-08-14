# -*- coding: utf-8 -*-

import numpy as np

from arpym.statistics.simulate_normal import simulate_normal


def simulate_garch(x_tnow, x_tnow_prec, param, j_, m_, sig2_tnow=None):
    """For details, see here.

    Parameters
    ----------
        x_t_now : float
        x_t_now_prec : float
        param : list or array, shape(4,)
        m_ : int
        j_ : int
        sig2_t_now : scalar, optional

    Returns
    -------
        x_tnow_thor : array, shape(j_, m_ + 1)
        sig2_tnow_thor : array, shape(j_, m_ + 1)

    """
    # check stationarity and initialize
    a, b, c, mu = param

    if sig2_tnow is None:
        if a+b >= 1:
            print("GARCH process is not stationary!" +
                  " Please provide initial variance.")
            return 0, 0
        sig2_t_now = c / (1. - a - b)

    # Step 0: calculate risk driver increment

    dx_t_now = x_tnow - x_tnow_prec

    # Monte Carlo scenarios for the path of GARCH
    sig2_tnow_thor = np.full((j_, m_ + 1), sig2_t_now)  # variances
    dx_t = np.full((j_, m_ + 1), dx_t_now)  # GARCH process

    # Step1: Increment process

    for m in range(1, m_ + 1):
        # generate j_ scenarios of for the shocks
        epsi = simulate_normal(0, 1, j_).reshape(-1)
        # compute scenarios for the variance
        sig2_tnow_thor[:, m] = c + b*sig2_tnow_thor[:, m-1] + \
            a*(dx_t[:, m-1]-mu)**2
        # compute scenarios for the risk driver increment
        dx_t[:, m] = mu + np.sqrt(sig2_tnow_thor[:, m]) * epsi

    # Step 2: Monte Carlo scenarios for the path of the GARCH(1,1)

    x_tnow_thor = np.cumsum(dx_t, axis=1) + x_tnow_prec

    return np.squeeze(x_tnow_thor), np.squeeze(sig2_tnow_thor)
