# -*- coding: utf-8 -*-

import numpy as np

from arpym.statistics.twist_scenarios_mom_match import \
    twist_scenarios_mom_match


def simulate_normal(mu_, sigma2_, j_, method='Riccati', d=None):
    """For details, see here.

    Parameters
    ----------
        mu_ : array, shape (n_,)
        sigma2_ : array, shape (n_,n_)
        j_ : int
        method : string, optional
        d : array, shape (k_,n_), optional

    Returns
    -------
        x : array, shape: (j_,n_) if n_>1 or (j_,) for n_=1

    """

    if np.ndim(mu_) >= 1:
        n_ = len(mu_)
    else:
        n_ = 1

    mu_ = np.reshape(mu_, n_).copy()
    sigma2_ = np.reshape(sigma2_, (n_, n_)).copy()

    if j_ == 1:
        if n_ > 1:
            x_ = np.random.multivariate_normal(mu_, sigma2_, j_)
            return np.squeeze(x_)
        else:
            x_ = np.random.normal(mu_[0], np.sqrt(sigma2_)[0, 0],
                                  j_).reshape(-1, 1)
            return np.squeeze(x_)

    if j_ < n_*(n_+1)/2+n_:
        x_ = np.random.multivariate_normal(mu_, sigma2_, j_)
        return np.squeeze(x_)

    # Step 1: Standard normal MC scenarios
    j_half = int(np.ceil(j_/2.))

    # generate standard normal Monte Carlo scenarios
    x_check = np.random.randn(j_half, n_)

    # Step 2: Antithetic scenarios
    # store first j_half scenarios, generate and store antithetical scenarios
    x = np.concatenate((x_check, -x_check))[:j_, :]

    # Step 3: Twisted normal MC scenarios
    x_ = twist_scenarios_mom_match(x, mu_, sigma2_, method=method, d=d)

    return np.squeeze(x_)
