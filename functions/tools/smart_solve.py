# -*- coding: utf-8 -*-

import numpy as np

from arpym.estimation.factor_analysis_paf import factor_analysis_paf


def smart_solve(sigma2, y=None, max_cond=1e15, k_=None):
    """For details, see here.

    Parameters
    ----------
        sigma2 : array, shape (n_, n_)
        y : array, shape (n_, m_), optional
        max_cond : float, optional
        k_ : integer, optional

    Returns
    -------
        x : array, shape (n_, m_)

    """
    n_ = sigma2.shape[0]

    if y is None:
        y = np.eye(sigma2.shape[0])

    if k_ is None:
        k_ = int(max(1., 0.1 * n_))

    if np.linalg.cond(sigma2) < max_cond:
        x = np.linalg.solve(sigma2, y)
    else:
        beta, delta2 = factor_analysis_paf(sigma2, k_, maxiter=100, eps=1e-4)

        # binomial inverse theorem
        rho2 = beta @ np.linalg.solve((beta.T / delta2) @
                                      beta + np.eye(k_), beta.T)
        x = (y.T / delta2 - (y.T / delta2) @ rho2 / delta2).T

    return x
