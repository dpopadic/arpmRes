# -*- coding: utf-8 -*-

import numpy as np

from arpym.estimation.cov_2_corr import cov_2_corr
from arpym.estimation.factor_analysis_paf import factor_analysis_paf
from arpym.statistics.simulate_normal import simulate_normal


def simulate_normal_dimred(mu, sigma2, j_, k_):
    """For details, see here.

    Parameters
    ----------
        mu : array, shape (n_,)
        sigma2 : array, shape (n_,n_)
        j_ : int
        k_ : int

    Returns
    -------
        x : array, shape (j_,n_) if n_>1 or (j_,) for n_=1
        beta : array, shape (n_,k) if k_>1 or (n_,) for k_=1

    """

    if np.ndim(mu) == 1:
        mu = np.array(mu).reshape(-1).copy()
        sigma2 = np.array(sigma2)
        n_ = len(mu)
    else:
        n_ = 1

    mu = np.reshape(mu, n_)
    sigma2 = np.reshape(sigma2, (n_, n_))

    # Step 1. Correlation
    rho2, _ = cov_2_corr(sigma2)

    # Step 2. Factor analysis
    beta, delta2 = factor_analysis_paf(rho2, k_)
    delta = np.sqrt(delta2)

    # Step 4. Systematic scenarios
    z_tilde = simulate_normal(np.zeros(k_), np.eye(k_), j_).reshape(-1, k_)

    # Step 5. Idiosyncratic scenarios
    u_tilde = simulate_normal(np.zeros(n_), np.eye(n_), j_).reshape(j_, n_)

    # Step 6. Output
    x = mu + (z_tilde @ np.atleast_2d(beta.T) + u_tilde @ np.diag(delta)) \
        @ np.diag(np.sqrt(np.diag(sigma2)))

    return np.squeeze(x), np.squeeze(beta)
