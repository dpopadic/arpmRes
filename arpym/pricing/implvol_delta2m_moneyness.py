#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import interpolate
from scipy.stats import norm


def implvol_delta2m_moneyness(sigma_delta, tau, delta_moneyness, y, tau_y, l_):
    """For details, see here.

    Parameters
    ----------
        sigma_delta : array, shape (t_, k_, n_)
        tau : array, shape (k_,)
        delta_moneyness : array, shape (n_,)
        y : array, shape (t_, d_)
        tau_y : array, shape (d_,)
        l_ : scalar

    Returns
    -------
       sigma_m : array, shape (t_, j_, l_)
       m_moneyness : array, shape (l_,)

    """

    # Step 1: Convert delta-moneyness into m-moneyness

    t_ = sigma_delta.shape[0]
    k_ = len(tau)
    n_ = len(delta_moneyness)
    y_grid_t = np.zeros((t_, k_, n_))
    m_data = np.zeros((t_, k_, n_))
    tau_y = np.atleast_1d(tau_y)
    for t in range(t_):
        if tau_y.shape[0] == 1:
            y_grid_t[t, :, :] = np.tile(np.atleast_2d(y).T,
                                        (1, n_))
        else:
            y_grid_tmp = interpolate.interp1d(tau_y, y[t, :])
            y_grid_t[t, :, :] = np.tile(np.atleast_2d(y_grid_tmp(tau)).T,
                                        (1, n_))
        m_data[t, :, :] = norm.ppf(delta_moneyness) * sigma_delta[t, :, :] - \
                          ((y_grid_t[t, :, :] + sigma_delta[t, :, :] ** 2 / 2).T * np.sqrt(tau)).T

    # Step 2: Construct m_moneyness grid

    # min m-moneyness
    min_m = np.min(m_data)
    # max m-moneyness
    max_m = np.max(m_data)
    # equally-spaced grid between minimal and maximal m-moneyness
    m_moneyness = min_m + (max_m - min_m) * np.arange(l_) / (l_ - 1)

    # Step 3: Implied volatility surface in m-moneyness

    sigma_m = np.zeros((t_, k_, l_))
    for t in range(t_):
        for k in range(k_):
            poly_coef = np.polyfit(m_data[t, k, :], sigma_delta[t, k, :], 2)
            polyf = np.poly1d(poly_coef)
            sigma_m[t, k, :] = polyf(m_moneyness.flatten())

    return sigma_m, np.squeeze(m_moneyness)
