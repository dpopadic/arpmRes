# -*- coding: utf-8 -*-
import numpy as np
import warnings


def marchenko_pastur(q, ll, sigma2=1):
    """For details, see here.

    Parameters
    ----------
         q : scalar
         ll : scalar
         sigma2 : scalar

    Returns
    -------
         x : array, shape (ll_, )
         y : array, shape (ll_, )
         xlim : array, shape (2, )

    """

    eps = 1e-9

    if sigma2 < 10 * eps:  # if sigma2 is too small, push it above a threshold
        sigma2 = 10 * eps

    xlim = np.array([(1 - 1 / np.sqrt(q)) ** 2, (1 + 1 / np.sqrt(q)) ** 2]) * sigma2
    xlim_tmp = [0, 0]
    if q > 1:
        xlim_tmp[1] = xlim[1] - eps
        xlim_tmp[0] = xlim[0] + eps
        dx = (xlim_tmp[1] - xlim_tmp[0]) / (ll - 1)
        x = xlim_tmp[0] + dx * np.arange(ll)
        y = q * np.sqrt((4 * x) / (sigma2 * q) - (x / sigma2 + 1 / q - 1) ** 2) / (2 * np.pi * x)
    elif q < 1:
        xlim_tmp[1] = xlim[1] - eps
        xlim_tmp[0] = xlim[0] + eps
        dx = (xlim_tmp[1] - xlim_tmp[0]) / (ll - 2)
        x = xlim_tmp[0] + dx * np.arange(ll - 1)
        y = q * np.sqrt((4 * x) / (sigma2 * q) - (x / sigma2 + 1 / q - 1) ** 2) / (2 * np.pi * x)
        xlim[0] = 0
        x = [0, x]
        y = [(1 - q), y]
    else:
        xlim = np.array([0, 4]) * sigma2
        dx = xlim[1] / ll
        x = dx * np.arange(1, ll)
        y = np.sqrt(4 * x / sigma2 - (x / sigma2) ** 2) / (2 * np.pi * x)

    return x, y, xlim
