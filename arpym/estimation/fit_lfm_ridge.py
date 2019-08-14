# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import Ridge

from arpym.statistics.meancov_sp import meancov_sp


def fit_lfm_ridge(x, z, p=None, lam=1e-4, fit_intercept=True):
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

    if fit_intercept is True:
        m_x = p @ x
        m_z = p @ z
    else:
        m_x = np.zeros(n_,)
        m_z = np.zeros(k_,)

    x_p = ((x - m_x).T * np.sqrt(p)).T
    z_p = ((z - m_z).T * np.sqrt(p)).T

    clf = Ridge(alpha=lam, fit_intercept=False)
    clf.fit(z_p, x_p, sample_weight=p)

    beta = clf.coef_

    if k_ == 1:
        alpha = m_x - beta * m_z
        u = x - alpha - z * beta
    else:
        alpha = m_x - beta @ m_z
        u = x - alpha - z @ beta.T

    _, s2_u = meancov_sp(u, p)

    return alpha, beta, s2_u, u
