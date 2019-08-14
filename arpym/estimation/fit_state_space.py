#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from arpym.estimation.fit_factor_analysis import fit_factor_analysis
from arpym.estimation.fit_lfm_ols import fit_lfm_ols
from arpym.statistics.kalman_filter import kalman_filter


def fit_state_space(x, k_, p=None):
    """For details, see here.

    Parameters
    ----------
        x : array, shape (t_, n_) if n_>1 or (t_, ) for n_=1
        p : array, shape (t_,)
        k_ : scalar

    Returns
    -------
        h : array, shape (t_, k_) if k_>1 or (t_, ) for k_=1
        alpha_hat : array, shape (n_,)
        beta_hat : array, shape (n_, k_) if k_>1 or (n_, ) for k_=1
        delta2_hat : array, shape(n_, n_)
        alpha_hat_h : array, shape(k_,)
        beta_hat_h : array, shape(k_, k_)
        sigma2_hat_h : array, shape(k_, k_)

    """

    if p is None:
        t_= x.shape[0]
        if len(x.shape) == 1:
            n_ = 1
        else:
            n_ = x.shape[1]
        p = np.ones(t_) / t_  # equal probabilities as default value

    # Step 1: Estimation of statistical LFM

    alpha_hat, beta_hat, delta2_hat, h_fa = fit_factor_analysis(x, k_, p,
                                                                'MLF')
    if np.ndim(beta_hat)==0:
        beta_hat = np.atleast_1d(beta_hat)
    if len(beta_hat.shape) == 1:
        beta_hat = beta_hat.reshape(-1, 1)

    # Step 2: Estimation of regression LFM

    alpha_hat_h, beta_hat_h, sigma2_hat_h, _ = fit_lfm_ols(h_fa[1:, ],
                                                           h_fa[:-1, ],
                                                           p[:-1])
    alpha_hat_h, beta_hat_h, sigma2_hat_h =\
        np.atleast_1d(alpha_hat_h), np.atleast_2d(beta_hat_h), np.atleast_2d(sigma2_hat_h)

    # Step 3: Extraction of hidden factors

    h = kalman_filter(x, alpha_hat, beta_hat, np.diagflat(delta2_hat),
                      alpha_hat_h, beta_hat_h, sigma2_hat_h)
    return h, np.squeeze(alpha_hat), np.squeeze(beta_hat), np.squeeze(delta2_hat),\
        np.squeeze(alpha_hat_h), np.squeeze(beta_hat_h), np.squeeze(sigma2_hat_h)
