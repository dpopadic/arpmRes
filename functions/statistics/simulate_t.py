# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from arpym.statistics.simulate_normal import simulate_normal
from arpym.tools.transpose_square_root import transpose_square_root
from arpym.statistics.twist_scenarios_mom_match import twist_scenarios_mom_match


def simulate_t(mu, sigma2, nu, j_, stoc_rep=None, method='Riccati', d=None):
    """For details, see here.

    Parameters
    ----------
        mu : array, shape (n_,)
        sigma2 : array, shape (n_,n_)
        nu : int
        j_ : int
        stoc_rep : bool, optional
        method : string, optional
        d : array, shape (k_,n_), optional

    Returns
    -------
        x : array, shape(j_,n_) if n_>1 or (j_,) for n_=1

    """

    if isinstance(mu, (list, tuple, np.ndarray)):
        mu = np.array(mu)
        sigma2 = np.array(sigma2)
        n_ = len(mu)
    else:
        n_ = 1
        mu = np.reshape(mu, n_)
        sigma2 = np.reshape(sigma2, (n_, n_))

    if stoc_rep is None:
        stoc_rep = False

    if stoc_rep is False:
        # Step 1: Riccati root
        sigma = transpose_square_root(sigma2, method, d)

        # Step 2: Radial scenarios
        r = np.sqrt(n_ * sp.stats.f.ppf(np.random.rand(j_, 1), n_, nu))

        # Step 3: Normal scenarios
        n = simulate_normal(np.zeros(n_), np.eye(n_), j_).reshape(-1, n_)

        # Step 4: Uniform component
        normalizers = np.linalg.norm(n, axis=1)
        y = n / normalizers[:, None]

        # Step 5: Output
        x = mu + r * y @ sigma

    else:  # Alternative stochastic representation
        # Step 6: Normal scenarios
        n = simulate_normal(np.zeros(n_), sigma2, j_, method, d).reshape(-1, n_)

        # Step 7: Chi-squared scenarios
        v = sp.stats.chi2.ppf(np.random.rand(j_, 1), nu)

        # Step 8: Output
        x = mu + n / np.sqrt(v / nu)

    return np.squeeze(x)
