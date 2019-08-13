# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp

from sklearn.metrics.pairwise import laplacian_kernel


def min_corr_toeplitz(c2, tau=None, gamma0=1.):
    """For details, see here.

    Parameters
    ----------
        c2 : array, shape (n_, n_)
        tau : array, shape (n,), optional
        g0 : float, optional

    Returns
    -------
        c2_star : array, shape (n_, n_)
        gamma_star : array, shape (n_,)

    """

    n_ = c2.shape[0]
    if tau is None:
        tau = np.array(range(n_))
    tau = tau.reshape(n_, 1)

    # Step 1: Compute the square Frobenius norm between two correlations

    def func(g):
        return np.linalg.norm(laplacian_kernel(tau, tau, g) - c2, ord='f')

    # Step 2: Calibrate the parameter gamma

    gamma_star = sp.optimize.minimize(func, gamma0, bounds=[(0, None)])['x'][0]

    # Step 3: Compute the Toeplitz correlation

    c2_star = laplacian_kernel(tau, tau, gamma_star)

    return c2_star, gamma_star
