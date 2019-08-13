# -*- coding: utf-8 -*-
import numpy as np


def bootstrap_hfp(epsi, p=None, j_=1000):
    """For details, see here.

    Parameters
    ----------
        epsi : array, shape (t_, i_)
        p : array, shape (t_, )
        j_ : int

    Returns
    -------
        epsi_j : array, shape(j_, i_)

    """

    if len(epsi.shape) == 1:
        epsi = epsi.reshape(-1, 1)
    t_, i_ = epsi.shape

    if p is None:
        p = np.ones(t_) / t_

    # Step 1: Compute subintervals

    s = np.r_[np.array([0]), np.cumsum(p)]

    # Step 2: Draw scenarios from uniform distribution

    u = np.random.rand(j_)

    # Step 3: Compute invariant scenarios

    ind = np.digitize(u, bins=s) - 1
    epsi_j = epsi[ind, :]

    return epsi_j
