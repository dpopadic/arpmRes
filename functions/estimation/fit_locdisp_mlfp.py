# -*- coding: utf-8 -*-

import numpy as np

from arpym.statistics.meancov_sp import meancov_sp
from arpym.statistics.mvt_logpdf import mvt_logpdf
from arpym.tools.mahalanobis_dist import mahalanobis_dist


def fit_locdisp_mlfp(epsi, *, p=None, nu=1000, threshold=1e-3,
                  maxiter=1000, print_iter=False):
    """For details, see here.

    Parameters
    ----------
        epsi : array, shape (t_, i_)
        p : array, shape (t_,), optional
        nu: float, optional
        threshold : float, optional
        maxiter : int, optional
        print_iter : bool

    Returns
    -------
        mu : array, shape (i_,)
        sigma2 : array, shape (i_, i_)

    """

    if len(epsi.shape) == 1:
        epsi = epsi.reshape(-1, 1)

    t_, i_ = epsi.shape

    if p is None:
        p = np.ones(t_) / t_

    # Step 0: Set initial values using method of moments

    mu, sigma2 = meancov_sp(epsi, p)

    if nu > 2.:
        # if nu <=2, then the covariance is not defined
        sigma2 = (nu - 2.) / nu * sigma2

    for i in range(maxiter):

        # Step 1: Update the weights

        if nu >= 1e3 and np.linalg.det(sigma2) < 1e-13:
            w = np.ones(t_)
        else:
            w = (nu + i_) / (nu + mahalanobis_dist(epsi, mu, sigma2) ** 2)
        q = w * p

        # Step 2: Update location and dispersion parameters

        mu_old, sigma2_old = mu, sigma2
        mu, sigma2 = meancov_sp(epsi, q)
        mu = mu / np.sum(q)

        # Step 3: Check convergence

        err = max(np.linalg.norm(mu - mu_old, ord=np.inf) /
                  np.linalg.norm(mu_old, ord=np.inf),
                  np.linalg.norm(sigma2 - sigma2_old, ord=np.inf) /
                  np.linalg.norm(sigma2_old, ord=np.inf))

        if print_iter is True:
            print('Iter: %i; Loglikelihood: %.5f; Error: %.5f' %
                  (i, p @ mvt_logpdf(epsi, mu, sigma2, nu), err))

        if err <= threshold:
            break
    return np.squeeze(mu), np.squeeze(sigma2)
