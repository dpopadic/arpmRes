#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def normal_canonical(mu, sig2):
    """For details, see here.

    Parameters
    ----------
        mu: array, shape(n_,)
        sig2: array, shape(n_, n_)

    Returns
    -------
        theta_mu: array, shape(n_,)
        theta_sig2: array, shape(n_, n_)


    """
    if np.ndim(mu) == 0:
        mu = np.atleast_1d(mu)
    if np.ndim(sig2) < 2:
        sig2 = np.atleast_2d(sig2)
    # compute expectation
    theta_mu = np.linalg.solve(sig2, mu)
    # compute covariance
    n_ = sig2.shape[0]
    theta_sig = -1/2 * np.linalg.solve(sig2, np.eye(n_))
    return np.squeeze(theta_mu), np.squeeze(theta_sig)
