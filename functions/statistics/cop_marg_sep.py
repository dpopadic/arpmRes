# -*- coding: utf-8 -*-

import numpy as np
from arpym.statistics.cdf_sp import cdf_sp


def cop_marg_sep(x, p=None):
    """For details, see here.

    Parameters
    ----------
        x : array, shape (j_, n_) if n_>1 or (j_,) for n_=1
        p : array, optional, shape (j_,)

    Returns
    -------
        u : array, shape (j_, n_) if n_>1 or (j_,) for n_=1
        x_grid : array, shape (j_, n_) if n_>1 or (j_,) for n_=1
        cdf_x : array, shape (j_, n_) if n_>1 or (j_,) for n_=1

    """

    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    j_, n_ = x.shape
    if p is None:
        p = np.ones(j_) / j_  # equal probabilities as default value

    # Step 1: Sort scenarios

    x_grid, ind_sort = np.sort(x, axis=0), np.argsort(x, axis=0)  # sorted scenarios

    # Step 2: Marginal cdf's

    cdf_x = np.zeros((j_, n_))
    for n in range(n_):
        cdf_x[:, n] = cdf_sp(x_grid[:, n], x[:, n], p)

    # Step 3: Copula scenarios

    u = np.zeros((j_, n_))
    for n in range(n_):
        u[ind_sort[:, n], n] = cdf_x[:, n]

    u[u >= 1] = 1 - np.spacing(1)
    u[u <= 0] = np.spacing(1)  # clear spurious outputs

    return np.squeeze(u), np.squeeze(x_grid), np.squeeze(cdf_x)
