#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def gram_schmidt(sigma2):
    """For details, see here.

    Parameters
    ----------
        sigma2 : array, shape (n_,n_)

    Returns
    -------
        g : array, shape (n_,n_)

    """
    n_ = sigma2.shape[0]

    # Step 0. Initialization
    g = np.empty_like(sigma2)
    v = np.zeros((n_, n_-1))
    a = np.eye(n_)

    for n in range(n_):
        a_n = a[:, [n]]
        for m in range(n):

            # Step 1. Projection
            v[:, [m]] = (g[:, [m]].T @ sigma2 @ a_n) * g[:, [m]]

        # Step 2. Orthogonalization
        u_n = a_n - v[:, :n].sum(axis=1).reshape(-1, 1)

        # Step 3. Normalization
        g[:, [n]] = u_n/np.sqrt(u_n.T @ sigma2 @ u_n)

    return g
