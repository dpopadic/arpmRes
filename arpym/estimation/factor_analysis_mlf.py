# -*- coding: utf-8 -*-

import numpy as np

from arpym.estimation.factor_analysis_paf import factor_analysis_paf


def factor_analysis_mlf(sigma2, k_=None, b=None, v=None,
                        maxiter=100, eps=1e-2):
    """For details, see here.

    Parameters
    ----------
        sigma2 : array, shape (n_, n_)
        k_ : integer, optional
        b : array, shape (n_, k_), optional
        v : array, shape (n_,), optional
        maxiter : integer, optional
        eps : float, optional

    Returns
    -------
        beta_mlf :  array, shape (n_, k_) for k_>1 or (n_, ) for k_=1
        delta2_mlf : array, shape (n_,)

    """

    # Step 0: Initialize parameters

    # Set default values
    n_ = sigma2.shape[0]
    if (k_ is None):
        k_ = int(n_/2.0)
    if b is None or v is None:
        b, v = factor_analysis_paf(sigma2, k_, maxiter=1)
    if np.ndim(b)==0:
        b = np.atleast_1d(b)
    if len(b.shape) == 1:
        b = b.reshape(-1, 1)

    for i in range(maxiter):
        v_prev = v
        b_prev = b

        # Step 1: Compute matrix eta

        eta = np.linalg.inv(b@b.T + np.diagflat(v))@b

        # Step 2: Update factor loadings

        b = sigma2 @ eta @ np.linalg.inv(np.eye(k_) - b.T @ eta +
                                         eta.T @ sigma2 @ eta)

        # Step 3: Update residual variances

        v = np.diag(sigma2 - sigma2@eta@b.T)
        if np.any(v < 0):
            gamma = min(np.abs(np.diag(sigma2)[v < 0] /
                               np.diag(b@eta.T@sigma2)[v < 0]))
            b = gamma*b
            v = np.diag(sigma2 - sigma2@eta@b.T)

        # Step 4: Check convergence

        err = max(np.max(np.abs(v - v_prev))/max(np.max(np.abs(v_prev)),
                  1e-20),
                  np.max(np.abs(b - b_prev))/max(np.max(np.abs(b_prev)),
                                                 1e-20))

        if err < eps:
            break

    delta2_mlf = v
    beta_mlf = b

    return np.squeeze(beta_mlf), delta2_mlf
