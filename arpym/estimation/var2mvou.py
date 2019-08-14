# -*- coding: utf-8 -*-

import numpy as np
import warnings


def var2mvou(b_dt, mu_dt, sig2_dt, dt):
    """For details, see here.

    Parameters
    ----------
        b_dt : array, shape(d_, d_)
        mu_dt : array, shape(d_, )
        sig2_dt : array, shape(d_, d_)
        dt : scalar

    Returns
    -------
        theta : array, shape(d_, d_)
        mu_mvou : array, shape(d_)
        sig2_mvou : array, shape(d_, d_)

    """

    b_dt = np.atleast_2d(b_dt).copy()
    mu_dt = np.atleast_1d(mu_dt).copy()
    sig2_dt = np.atleast_2d(sig2_dt).copy()
    d_ = mu_dt.shape[0]

    # Step 1: Compute theta parameter of the embedding MVOU

    # compute eigenvalues and eigenvectors of b_dt
    lambda2, e = np.linalg.eig(b_dt)

    # warning: eigvals with negative real part, explosive MVOU
    if any(np.real(np.log(lambda2)) > 0):
        warnings.warn(('Warning: eigenvalues with ' +
                       'negative part: the MVOU is explosive'))

    theta = -(1/dt)*e@np.diag(np.log(lambda2))@np.linalg.pinv(e)
    theta = np.real(theta)

    # Step 2: Compute the drift of the embedding MVOU

    mu_mvou = theta@np.linalg.pinv(np.eye(d_)-b_dt)@mu_dt

    # Step 3: Compute the covariance matrix of the embedding MVOU

    vec_sig2 = np.reshape(sig2_dt, (d_ ** 2, 1), 'F')
    l_kprod, e_kprod = np.linalg.eig(np.kron(b_dt, b_dt))
    ln_bkronb = e_kprod@np.diag(np.log(l_kprod))@np.linalg.pinv(e_kprod)

    sig2_mvou = -(1/dt)*np.linalg.pinv(np.eye(d_**2)-np.kron(b_dt, b_dt)) @ \
        ln_bkronb@vec_sig2
    sig2_mvou = np.real(np.reshape(sig2_mvou, (d_, d_), 'F'))

    return theta, mu_mvou.reshape(-1), sig2_mvou
