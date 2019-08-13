# -*- coding: utf-8 -*-

import numpy as np
from arpym.statistics.quantile_sp import quantile_sp


def cop_marg_comb(u, x_sort, cdf_x=None):
    """For details, see here.

    Parameters
    ----------
       u : array, shape (j_, n_) if n_>1 or (j_,) for n_=1
       x_sort : array, shape (j_, n_) if n_>1 or (j_,) for n_=1
       cdf_x : array, shape (j_, n_) if n_>1 or (j_,) for n_=1

    Returns
    -------
        x : array, shape (j_, n_) if n_>1 or (j_,) for n_=1

    """
    if len(x_sort.shape) == 1:
        x_sort = x_sort.reshape(-1, 1)
    if len(u.shape) == 1:
        u = u.reshape(-1, 1)

    # Step 1: Extract probabilities

    if cdf_x is None:
        j_, n_ = x_sort.shape
        p_sort = np.zeros((j_, n_))
        for n in range(n_):
            p_sort[:, n] = np.ones(j_) / j_  # equal probabilities as default value
    else:
        j_, n_ = x_sort.shape
        p_sort = np.zeros((j_, n_))
        for n in range(n_):
            p_sort[:, n] = np.append(cdf_x[0, n], np.diff(cdf_x[:, n]))

    # Step 2: Copula-marginal scenarios

    x = np.zeros((j_, n_))
    for n in range(n_-1):
            x[:, n] = quantile_sp(u[:, n], x_sort[:, n], p=p_sort[:, n])

    return np.squeeze(x)
