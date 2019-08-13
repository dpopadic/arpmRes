#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.optimize import minimize

from arpym.estimation.fit_locdisp_mlfp import fit_locdisp_mlfp
from arpym.statistics.mvt_logpdf import mvt_logpdf


def fit_t_dof(x, p=None, lb=2., ub=10.):
    """For details, see here.

    Parameters
    ----------
        x : array, shape (t_, i_)
        p : array, shape (t_,), optional
        lb : float
        ub : float

    Returns
    -------
        nu : float
        mu : array, shape (i_,)
        sigma2 : array, shape (i_, i_)

    """

    # Step 1: Compute negative log-likelihood function
    def llh(nu):
        mu, sigma2 = fit_locdisp_mlfp(x, p, nu, maxiter=200)
        mu, sigma2 = np.atleast_1d(mu), np.atleast_2d(sigma2)
        return -p @ mvt_logpdf(x, mu, sigma2, nu)

    # Step 2: Find the optimal dof
    nu = minimize(llh, 5., bounds=[(lb, ub)])['x']
    mu, sigma2 = fit_locdisp_mlfp(x, p, nu, maxiter=200)
    mu, sigma2 = np.atleast_1d(mu), np.atleast_2d(sigma2)

    return nu, mu, sigma2
