# -*- coding: utf-8 -*-

import numpy as np
from sklearn.covariance import graphical_lasso

from arpym.estimation.fit_lfm_lasso import fit_lfm_lasso
from arpym.tools.mahalanobis_dist import mahalanobis_dist
from arpym.statistics.mvt_logpdf import mvt_logpdf
from arpym.statistics.meancov_sp import meancov_sp


def fit_lfm_roblasso(x, z, p=None, nu=1e9, lambda_beta=0., lambda_phi=0.,
                     tol=1e-3, fit_intercept=True, maxiter=500,
                     print_iter=False, rescale=False):
    """For details, see here.

    Parameters
    ----------
        x : array, shape(t_, n_)
        z : array, shape(t_, k_)
        p : array, optional, shape(t_)
        nu : scalar, optional
        lambda_beta : scalar, optional
        lambda_phil : scalar, optional
        tol : float, optional
        fit_intercept: bool, optional
        maxiter : scalar, optional
        print_iter : bool, optional
        rescale : bool, optional

    Returns
    -------
       alpha_RMLFP : array, shape(n_,)
       beta_RMLFP : array, shape(n_,k_)
       sig2_RMLFP : array, shape(n_,n_)

    """

    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

    if len(z.shape) == 1:
        z = z.reshape(-1, 1)

    t_, n_ = x.shape

    if p is None:
        p = np.ones(t_) / t_

    # rescale the variables
    if rescale is True:
        _, sigma2_x = meancov_sp(x, p)
        sigma_x = np.sqrt(np.diag(sigma2_x))
        x = x / sigma_x

        _, sigma2_z = meancov_sp(z, p)
        sigma_z = np.sqrt(np.diag(sigma2_z))
        z = z / sigma_z

    # Step 0: Set initial values using method of moments

    alpha, beta, sigma2, u = fit_lfm_lasso(x, z, p, lambda_beta,
                                           fit_intercept=fit_intercept)
    mu_u = np.zeros(n_)

    for i in range(maxiter):

        # Step 1: Update the weights

        if nu >= 1e3 and np.linalg.det(sigma2) < 1e-13:
            w = np.ones(t_)
        else:
            w = (nu + n_) / (nu + mahalanobis_dist(u, mu_u, sigma2) ** 2)
        q = w * p
        q = q / np.sum(q)

        # Step 2: Update location and dispersion parameters

        alpha_old, beta_old = alpha, beta
        alpha, beta, sigma2, u = fit_lfm_lasso(x, z, q, lambda_beta,
                                               fit_intercept=fit_intercept)
        sigma2, _ = graphical_lasso((w @ p) * sigma2, lambda_phi)

        # Step 3: Check convergence

        errors = [np.linalg.norm(alpha - alpha_old, ord=np.inf) /
                  max(np.linalg.norm(alpha_old, ord=np.inf), 1e-20),
                  np.linalg.norm(beta - beta_old, ord=np.inf) /
                  max(np.linalg.norm(beta_old, ord=np.inf), 1e-20)]

        # print the loglikelihood and the error
        if print_iter is True:
            print('Iter: %i; Loglikelihood: %.5f; Errors: %.5f' %
                  (i, p @ mvt_logpdf(u, mu_u, sigma2, nu) -
                   lambda_beta * np.linalg.norm(beta, ord=1), max(errors)))

        if max(errors) <= tol:
            break

    if rescale is True:
        alpha = alpha * sigma_x
        beta = ((beta / sigma_z).T * sigma_x).T
        sigma2 = (sigma2.T * sigma_x).T * sigma_x

    return alpha, beta, sigma2
