#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from arpym.estimation.factor_analysis_paf import factor_analysis_paf
from arpym.estimation.factor_analysis_mlf import factor_analysis_mlf
from arpym.statistics.meancov_sp import meancov_sp


def fit_factor_analysis(x, k_, p=None, method='PrincAxFact'):
    """For details, see here.

    Parameters
    ----------
        x : array, shape (t_, n_) if n_>1 or (t_, ) for n_=1
        k_ : scalar
        p : array, shape (t_,), optional
        method : string, optional

    Returns
    -------
        alpha_hat : array, shape (n_,)
        beta_hat : array, shape (n_, k_) if k_>1 or (n_, ) for k_=1
        delta2 : array, shape(n_, n_)
        z_reg : array, shape(t_, n_) if n_>1 or (t_, ) for n_=1

    """
    t_ = x.shape[0]

    if len(x.shape) == 1:
        x = x.reshape((t_, 1))

    if p is None:
        p = np.ones(t_) / t_

    # Step 1: Compute HFP mean and covariance of X

    m_x_hat_hfp, s2_x_hat_hfp = meancov_sp(x, p)

    # Step 2: Estimate alpha

    alpha_hat = m_x_hat_hfp

    # Step 3: Decompose covariance matrix

    if method == 'PrincAxFact' or method.lower() == 'paf':
        beta_hat, delta2_hat = factor_analysis_paf(s2_x_hat_hfp, k_)
    else:
        beta_hat, delta2_hat = factor_analysis_mlf(s2_x_hat_hfp, k_)
    if k_ == 1:
        beta_hat = beta_hat.reshape(-1, 1)

    # Step 4: Compute factor analysis covariance matrix

    s2_x_hat_fa = beta_hat@beta_hat.T + np.diagflat(delta2_hat)

    # Step 5: Approximate hidden factor via regression

    if np.all(delta2_hat != 0):
        omega2 = np.diag(1/delta2_hat)
        z_reg = beta_hat.T @ \
            (omega2-omega2@beta_hat@
             np.linalg.inv(beta_hat.T@omega2@beta_hat + np.eye(k_))@
             beta_hat.T@omega2)@(x-m_x_hat_hfp).T
    else:
        z_reg = beta_hat.T@np.linalg.inv(s2_x_hat_fa)@(x-m_x_hat_hfp).T

    return alpha_hat, np.squeeze(beta_hat), delta2_hat, np.squeeze(z_reg.T)