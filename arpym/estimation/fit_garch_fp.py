# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize

from arpym.statistics.meancov_sp import meancov_sp
from arpym.estimation.exp_decay_fp import exp_decay_fp


def fit_garch_fp(dx, p=None, sigma2_0=None, param0=None,
                 g=0.95, rescale=False):
    """For details, see here.

    Parameters
    ----------
        dx : array, shape(t_,)
        p : array, optional, shape(t_)
        sigma2_0 : scalar, optional
        param0 : list or array, optional, shape(4,)
        g : scalar, optional
        rescale : bool, optional

    Returns
    -------
        param : list
        sigma2 : array, shape(t_,)
        epsi : array, shape(t_,)

    """

    t_ = dx.shape[0]

    # flexible probabilities
    if p is None:
        p = np.ones(t_) / t_

    # sample mean and variance
    if (param0 is None) or (rescale is True):
        m, s2 = meancov_sp(dx, p)

    if param0 is None:
        param0 = [0.01, g - 0.01, s2 * (1. - g), m]  # initial parameters

    # Step 0: Set default standard variance if not provided

    if sigma2_0 is None:
        p_tau = exp_decay_fp(t_, t_ / 3, 0)
        _, sigma2_0 = meancov_sp(dx, p_tau)

    # Step 1: standardize data if requested

    if rescale is True:
        dx = (dx - m) / np.sqrt(s2)
        param0[2] = param0[2] / s2
        param0[3] = param0[3] - m
        sigma2_0 = sigma2_0 / s2

    # Step 2: Compute negative log-likelihood of GARCH

    def theta(param):
        a, b, c, mu = param
        sigma2 = sigma2_0
        theta = 0.0
        for t in range(t_):
            # if statement added because of overflow when sigma2 is too low
            if np.abs(sigma2) > 1e-128:
                theta = theta + ((dx[t] - mu) ** 2 / sigma2
                                 + np.log(sigma2)) * p[t]
            sigma2 = c + a * (dx[t] - mu) ** 2 + b * sigma2

        return theta

    # Step 3: Minimize the negative log-likelihood

    # parameter boundaries
    bnds = ((1e-20, 1.), (1e-20, 1.), (1e-20, None), (None, None))
    # stationary constraints
    cons = {'type': 'ineq', 'fun': lambda param: g - param[0] - param[1]}
    a_hat, b_hat, c_hat, mu_hat = \
        minimize(theta, param0, bounds=bnds, constraints=cons)['x']

    # Step 4: Compute realized variance and invariants

    sigma2_hat = np.full(t_, sigma2_0)
    for t in range(t_ - 1):
        sigma2_hat[t + 1] = c_hat + a_hat * (dx[t] - mu_hat) ** 2 + \
                            b_hat * sigma2_hat[t]
    epsi = (dx - mu_hat) / np.sqrt(sigma2_hat)

    # Step 5: revert standardization at Step 1, if requested

    if rescale is True:
        c_hat = c_hat * s2
        mu_hat = mu_hat * np.sqrt(s2) + m
        sigma2_hat = sigma2_hat * s2

    return np.array([a_hat, b_hat, c_hat, mu_hat]), sigma2_hat, np.squeeze(epsi)
