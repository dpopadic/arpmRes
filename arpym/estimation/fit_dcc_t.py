# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

from arpym.estimation.cov_2_corr import cov_2_corr
from arpym.statistics.meancov_sp import meancov_sp


def fit_dcc_t(dx, p=None, *, rho2=None, param0=None, g=0.99):
    """For details, see here.

    Parameters
    ----------
        dx : array, shape(t_, i_)
        p : array, optional, shape(t_)
        rho2 : array, shape(i_, i_)
        param0 : list or array, shape(2,)
        g : scalar, optional

    Returns
    -------
        params : list, shape(3,)
        r2_t : array, shape(t_, i_, i_)
        epsi : array, shape(t_, i_)
        q2_t_ : array, shape(i_, i_)

    """

    # Step 0: Setup default values

    t_, i_ = dx.shape

    # flexible probabilities
    if p is None:
        p = np.ones(t_) / t_

    # target correlation
    if rho2 is None:
        _, rho2 = meancov_sp(dx, p)
        rho2, _ = cov_2_corr(rho2)

    # initial parameters
    if param0 is None:
        param0 = [0.01, g - 0.01]  # initial parameters

    # Step 1: Compute negative log-likelihood of GARCH

    def llh(params):
        a, b = params
        mu = np.zeros(i_)
        q2_t = rho2.copy()
        r2_t, _ = cov_2_corr(q2_t)
        llh = 0.0
        for t in range(t_):
            llh = llh - p[t] * multivariate_normal.logpdf(dx[t, :], mu, r2_t)
            q2_t = rho2 * (1 - a - b) + \
                a * np.outer(dx[t, :], dx[t, :]) + b * q2_t
            r2_t, _ = cov_2_corr(q2_t)

        return llh

    # Step 2: Minimize the negative log-likelihood

    # parameter boundaries
    bnds = ((1e-20, 1.), (1e-20, 1.))
    # stationary constraints
    cons = {'type': 'ineq', 'fun': lambda param: g - param[0] - param[1]}
    a, b = minimize(llh, param0, bounds=bnds, constraints=cons)['x']

    # Step 3: Compute realized correlations and residuals

    q2_t = rho2.copy()
    r2_t = np.zeros((t_, i_, i_))
    r2_t[0, :, :], _ = cov_2_corr(q2_t)

    for t in range(t_ - 1):
        q2_t = rho2 * (1 - a - b) + \
            a * np.outer(dx[t, :], dx[t, :]) + b * q2_t
        r2_t[t + 1, :, :], _ = cov_2_corr(q2_t)

    l_t = np.linalg.cholesky(r2_t)
    epsi = np.linalg.solve(l_t, dx)

    return [1. - a - b, a, b], r2_t, epsi, q2_t
