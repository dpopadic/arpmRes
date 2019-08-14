#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from arpym.statistics.meancov_sp import meancov_sp


def fit_lfm_ols(x_t, z_t, p_t=None, fit_intercept=True):
    """For details, see here.

    Parameters
    ----------
        x_t : array, shape (t_, n_) if n_>1 or (t_, ) for n_=1
        z_t : array, shape (t_, k_) if k_>1 or (t_, ) for k_=1
        p_t : array, optional, shape (t_,)
        fit_intercept : bool

    Returns
    -------
        alpha_hat_olsfp : array, shape (n_,)
        beta_hat_olsfp : array, shape (n_, k_) if k_>1 or (n_, ) for k_=1
        s2_u_hat_olsfp : array, shape (n_, n_)
        u_t : array, shape (t_, n_) if n_>1 or (t_, ) for n_=1

    """
    t_ = x_t.shape[0]

    if len(z_t.shape) < 2:
        z_t = z_t.reshape((t_, 1)).copy()
        k_ = 1
    else:
        k_ = z_t.shape[1]

    if len(x_t.shape) < 2:
        x_t = x_t.reshape((t_, 1)).copy()
        n_ = 1
    else:
        n_ = x_t.shape[1]

    if p_t is None:
        p_t = np.ones(t_) / t_

    # Step 1: Compute HFP mean and covariance of (X,Z)'

    if fit_intercept is True:
        m_xz_hat_hfp, s2_xz_hat_hfp = meancov_sp(np.c_[x_t, z_t], p_t)
    else:
        m_xz_hat_hfp = np.zeros(n_ + k_)
        s2_xz_hat_hfp = p_t*np.c_[x_t, z_t].T @ np.c_[x_t, z_t]

    # Step 2: Compute the OLSFP estimates

    s2_z_hat_hfp = s2_xz_hat_hfp[n_:, n_:]
    s_x_z_hat_hfp = s2_xz_hat_hfp[:n_, n_:]
    m_xz_hat_hfp = m_xz_hat_hfp.reshape(-1)
    m_z_hat_hfp = m_xz_hat_hfp[n_:].reshape(-1, 1)
    m_x_hat_hfp = m_xz_hat_hfp[:n_].reshape(-1, 1)

    beta_hat_olsfp = s_x_z_hat_hfp @ np.linalg.inv(s2_z_hat_hfp)
    alpha_hat_olsfp = m_x_hat_hfp - beta_hat_olsfp @ m_z_hat_hfp

    # Step 3: Compute residuals and OLSFP estimate of covariance of U

    u_t = (x_t.T - alpha_hat_olsfp - beta_hat_olsfp @ z_t.T).T
    _, s2_u_hat_olsfp = meancov_sp(u_t, p_t)

    return alpha_hat_olsfp[:, 0], np.squeeze(beta_hat_olsfp),\
        np.squeeze(s2_u_hat_olsfp), np.squeeze(u_t)
