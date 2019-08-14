# -*- coding: utf-8 -*-
import numpy as np


def min_rel_entropy_normal(mu_pri, s2_pri, v_mu, mu_view, v_sigma, sigma_view):
    """For details, see here.

    Parameters
    ----------
        mu_pri : array, shape (n_,)
        s2_pri : array, shape (n_, n_)
        v_mu : array, shape (k_, n_)
        mu_view : array, shape (k_,)
        v_sigma : array, shape (s_, n_)
        sigma_view : array, shape (s_, s_)

    Returns
    -------
        mu_pos : array, shape (n_,)
        s2_pos : array, shape (n_, n_)

    """

    s_, n_ = v_sigma.shape

    mu_pos = mu_pri + s2_pri@v_mu.T@np.linalg.solve(v_mu@s2_pri@v_mu.T,
                                                    mu_view - v_mu@mu_pri)
    a = np.linalg.solve(v_sigma@s2_pri@v_sigma.T, np.eye(s_))
    s2_pos = s2_pri + s2_pri@v_sigma.T@(a@sigma_view@a - a)@v_sigma@s2_pri

    return mu_pos, s2_pos
