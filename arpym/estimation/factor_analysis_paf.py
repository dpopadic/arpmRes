#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

from arpym.tools.pca_cov import pca_cov


def factor_analysis_paf(sigma2, k_=None, maxiter=100, eps=1e-2):
    """For details, see here.

    Parameters
    ----------
        sigma2 : array, shape (n_, n_)
        k : int, optional
        maxiter : integer, optional
        eps : float, optional

    Returns
    -------
        beta_paf : array, shape (n_, k_) for k_>1 or (n_, ) for k_=1
        delta2_paf : array, shape (n_,)

    """
    n_ = sigma2.shape[0]

    if k_ is None:
        k_ = int(n_/2.0)

    # Step 0: Initialize parameters

    v = np.zeros(n_)
    b = np.zeros((n_, k_))

    for i in range(maxiter):

        # Step 1: Compute the first k_ eigenvectors and eigenvalues

        e_k, lambda2_k = pca_cov(sigma2 - np.diagflat(v), k_)

        # Step 2: Compute the factor loadings and adjust if necessary

        # Step 2a
        b_new = e_k @ np.diag(np.sqrt(lambda2_k))
        # Step 2b
        idx = np.diag(b_new @ b_new.T) > np.diag(sigma2)
        if np.any(idx):
            b_new[idx, :] = np.sqrt(sigma2[idx, idx] /
                                    (b_new @ b_new.T)[idx,
                                                      idx]).reshape(-1, 1) *\
                            b_new[idx, :]

        # Step 3: Update residual variances

        v_new = np.diag(sigma2) - np.diag(b_new @ b_new.T)

        # Step 4: Check convergence

        # relative error
        err = max(np.max(np.abs(v_new - v))/max(np.max(np.abs(v)), 1e-20),
                  np.max(np.abs(b_new - b))/max(np.max(np.abs(b)), 1e-20))

        v = v_new
        b = b_new
        if err < eps:
            break

    beta_paf = b_new
    delta2_paf = v

    return np.squeeze(beta_paf), delta2_paf
