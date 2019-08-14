# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import lasso_path


def fit_lfm_lasso_path(x, z, p=None, lambdas=None, fit_intercept=True):
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

    _, coeff_, _ = lasso_path(z_p, x_p, alphas=lambdas/(2*t_),
                              fit_intercept=False)

    # lasso_path automatically sorts lambdas from the largest to the smallest,
    # so we have to revert order of coeff_ back to the original lambdas
    idx = np.argsort(lambdas)[::-1]
    betas = np.zeros((i_, n_, k_))
    betas[idx, :, :] = coeff_.transpose((2, 0, 1))
    alphas = m_x - betas @ m_z

    return alphas, betas
