# -*- coding: utf-8 -*-

import numpy as np

from arpym.estimation.fit_lfm_ols import fit_lfm_ols
from arpym.tools.mahalanobis_dist import mahalanobis_dist
from arpym.statistics.mvt_logpdf import mvt_logpdf
from arpym.statistics.meancov_sp import meancov_sp


def fit_lfm_mlfp(x, z, p=None, nu=4, tol=1e-3, fit_intercept=True,
                 maxiter=500, print_iter=False, rescale=False):
    """For details, see here.

    Parameters
    ----------
        x : array, shape (t_, n_) if n_>1 or (t_, ) for n_=1
        z : array, shape (t_, k_) if k_>1 or (t_, ) for k_=1
        p : array, optional, shape (t_,)
        nu : scalar, optional
        tol : float, optional
        fit_intercept: bool, optional
        maxiter : scalar, optional
        print_iter : bool, optional
        rescale : bool, optional

    Returns
    -------
       alpha : array, shape (n_,)
       beta : array, shape (n_, k_) if k_>1 or (n_, ) for k_=1
       sigma2 : array, shape (n_, n_)
       u : shape (t_, n_) if n_>1 or (t_, ) for n_=1

    """

    if np.ndim(x) < 2:
        x = x.reshape(-1, 1).copy()
    t_, n_ = x.shape
    if np.ndim(z) < 2:
        z = z.reshape(-1, 1).copy()
    t_, n_ = x.shape
    k_ = z.shape[1]

    if p is None:
        p = np.ones(t_) / t_

    # rescale the variables
    if rescale is True:
        _, sigma2_x = meancov_sp(x, p)
        sigma_x = np.sqrt(np.diag(sigma2_x))
        x = x.copy() / sigma_x

        _, sigma2_z = meancov_sp(z, p)
        sigma_z = np.sqrt(np.diag(sigma2_z))
        z = z.copy() / sigma_z

    # Step 0: Set initial values using method of moments

    alpha, beta, sigma2, u = fit_lfm_ols(x, z, p,
                                         fit_intercept=fit_intercept)
    alpha, beta, sigma2, u = \
        alpha.reshape((n_, 1)), beta.reshape((n_, k_)), \
        sigma2.reshape((n_, n_)), u.reshape((t_, n_))

    if nu > 2.:
        # if nu <=2, then the covariance is not defined
        sigma2 = (nu - 2.) / nu * sigma2

    mu_u = np.zeros(n_)

    for i in range(maxiter):

        # Step 1: Update the weights and historical flexible probabilities

        if nu >= 1e3 and np.linalg.det(sigma2) < 1e-13:
            w = np.ones(t_)
        else:
            w = (nu + n_) / (nu + mahalanobis_dist(u, mu_u, sigma2) ** 2)
        q = w * p
        q = q / np.sum(q)

        # Step 2: Update shift parameters, factor loadings and covariance

        alpha_old, beta_old, sigma2_old = alpha, beta, sigma2
        alpha, beta, sigma2, u = fit_lfm_ols(x, z, q,
                                             fit_intercept=fit_intercept)
        alpha, beta, sigma2, u = \
            alpha.reshape((n_, 1)), beta.reshape((n_, k_)), \
            sigma2.reshape((n_, n_)), u.reshape((t_, n_))
        sigma2 = (w @ p) * sigma2

        # Step 3: Check convergence
        beta_tilde_old = np.column_stack((alpha_old, beta_old))
        beta_tilde = np.column_stack((alpha, beta))
        errors = [np.linalg.norm(beta_tilde - beta_tilde_old, ord=np.inf) /
                  np.linalg.norm(beta_tilde_old, ord=np.inf),
                  np.linalg.norm(sigma2 - sigma2_old, ord=np.inf) /
                  np.linalg.norm(sigma2_old, ord=np.inf)]

        # print the loglikelihood and the error
        if print_iter is True:
            print('Iter: %i; Loglikelihood: %.5f; Errors: %.3e' %
                  (i, p @ mvt_logpdf(u, mu_u, sigma2, nu), max(errors)))

        if max(errors) < tol:
            break

    if rescale is True:
        alpha = alpha * sigma_x
        beta = ((beta / sigma_z).T * sigma_x).T
        sigma2 = (sigma2.T * sigma_x).T * sigma_x

    return np.squeeze(alpha), np.squeeze(beta), np.squeeze(sigma2), np.squeeze(u)
