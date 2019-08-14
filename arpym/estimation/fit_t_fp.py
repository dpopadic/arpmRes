# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize
from scipy.stats import t

from arpym.statistics.meancov_sp import meancov_sp


def fit_t_fp(x, p=None):
    """For details, see here.

    Parameters
    ----------
        x : array, shape (t_,)
        p : array, shape (t_,), optional

    Returns
    -------
        nu : float
        mu : float
        sigma2 : float

    """
    t_ = x.shape[0]
    if p is None:
        p = np.ones(t_) / t_

    # Step 1: Compute negative log-likelihood function
    def llh(params):
        nu, mu, sigma = params
        return -p @ t.logpdf(x, nu, mu, sigma)

    # Step 2: Find the optimal dof
    m, s2 = meancov_sp(x, p)
    param0 = [10, m, np.sqrt(s2)]
    bnds = [(1e-20, None), (None, None), (1e-20, None)]
    nu, mu, sigma = minimize(llh, param0, bounds=bnds)['x']
    sigma2 = sigma ** 2

    return nu, mu, sigma2
