# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import Lasso

from arpym.statistics.meancov_sp import meancov_sp
from arpym.estimation.fit_lfm_ols import fit_lfm_ols


def fit_lfm_lasso(x, z, p=None, lam=1e-2, fit_intercept=True):
    """For details, see here.

    Parameters
    ----------
        x : array, shape (t_, n_)
        z : array, shape (t_, k_)
        p : array, optional, shape (t_,)
        lam : float, optional
        fit_intercept : bool, optional

    Returns
    -------
        alpha : array, shape (n_,)
        beta : array, shape (n_, k_)
        s2_u : array, shape (n_, n_)
        u : array, shape (t_, n_)

    """

    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

    if len(z.shape) == 1:
        z = z.reshape(-1, 1)

    t_, n_ = x.shape
    k_ = z.shape[1]

    if p is None:
        p = np.ones(t_) / t_

    if lam == 0:
        alpha, beta, s2_u, u = fit_lfm_ols(x, z, p, fit_intercept)
    else:
        if fit_intercept is True:
            m_x = p @ x
            m_z = p @ z
        else:
            m_x = np.zeros(n_,)
            m_z = np.zeros(k_,)

        x_p = ((x - m_x).T * np.sqrt(p)).T
        z_p = ((z - m_z).T * np.sqrt(p)).T

        clf = Lasso(alpha=lam/(2.*t_), fit_intercept=False)
        clf.fit(z_p, x_p)
        beta = clf.coef_

        if k_ == 1:
            alpha = m_x - beta * m_z
            u = x - alpha - z * beta
        else:
            alpha = m_x - beta @ m_z
            u = x - alpha - z @ np.atleast_2d(beta).T

        _, s2_u = meancov_sp(u, p)

    return alpha, beta, s2_u, u
