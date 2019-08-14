# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import Ridge


def fit_lfm_ridge_path(x, z, p=None, lambdas=None, fit_intercept=True):
    """For details, see here.

    Parameters
    ----------
        x : array, shape (t_, n_)
        z : array, shape (t_, k_)
        p : array, optional, shape (t_,)
        lambdas : array, shape(i_,), optional
        fit_intercept : bool, optional

    Returns
    -------
        alpha : array, shape (i_, n_)
        beta : array, shape (i_, n_, k_)

    For details, see here.

    """

    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

    if len(z.shape) == 1:
        z = z.reshape(-1, 1)

    t_, n_ = x.shape
    k_ = z.shape[1]

    if lambdas is None:
        lambdas = np.array([0, 0.1, 0.2])
    i_ = lambdas.shape[0]

    if p is None:
        p = np.ones(t_) / t_

    if fit_intercept is True:
        m_x = p @ x
        m_z = p @ z
    else:
        m_x = np.zeros(n_,)
        m_z = np.zeros(k_,)

    x_p = ((x - m_x).T * np.sqrt(p)).T
    z_p = ((z - m_z).T * np.sqrt(p)).T

    # matches the number of targes to the number of lambdas
    x_p_multi = np.tile(x_p, (1, i_))
    lambdas_multi = np.tile(lambdas, (n_, 1)).T.reshape(n_ * i_)

    clf = Ridge(alpha=lambdas_multi, fit_intercept=False)
    clf.fit(z_p, x_p_multi)

    betas = clf.coef_.reshape((i_, n_, k_))
    alphas = m_x - betas @ m_z

    return alphas, betas
