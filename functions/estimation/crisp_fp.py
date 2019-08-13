# -*- coding: utf-8 -*-

import numpy as np
from arpym.statistics.cdf_sp import cdf_sp
from arpym.statistics.quantile_sp import quantile_sp


def crisp_fp(z, z_star, alpha):
    """For details, see here.

    Parameters
    ----------
        z : array, shape (t_, )
        z_star : array, shape (k_, )
        alpha : scalar

    Returns
    -------
        p : array, shape (t_,k_) if k_>1 or (t_,) for k_==1
        z_lb : array, shape (k_,)
        z_ub : array, shape (k_,)

    """

    z_star = np.atleast_1d(z_star)

    t_ = z.shape[0]
    k_ = z_star.shape[0]

    # Step 0: Compute cdf of the risk factor at target values

    cdf_z_star = cdf_sp(z_star, z, method="linear_interp")
    cdf_z_star = np.atleast_1d(cdf_z_star)

    # Step 1: Compute crisp probabilities

    z_lb = np.zeros(k_)
    z_ub = np.zeros(k_)
    p = np.zeros((k_, t_))
    pp = np.zeros((k_, t_))

    for k in range(k_):

        # compute range
        if z_star[k] <= quantile_sp(alpha/2, z,
                                    method="kernel_smoothing"):
            z_lb[k] = np.min(z)
            z_ub[k] = quantile_sp(alpha, z, method="kernel_smoothing")
        elif z_star[k] >= quantile_sp(1-alpha/2, z,
                                      method="kernel_smoothing"):
            z_lb[k] = quantile_sp(1-alpha, z, method="kernel_smoothing")
            z_ub[k] = np.max(z)
        else:
            z_lb[k] = quantile_sp(cdf_z_star[k]-alpha/2, z,
                                  method="kernel_smoothing")
            z_ub[k] = quantile_sp(cdf_z_star[k]+alpha/2, z,
                                  method="kernel_smoothing")

        # crisp probabilities
        pp[k, (z <= z_ub[k]) & (z >= z_lb[k])] = 1
        p[k, :] = pp[k, :]/np.sum(pp[k, :])

    return np.squeeze(p.T), np.squeeze(z_lb), np.squeeze(z_ub)
