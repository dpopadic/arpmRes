# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp


def pca_cov(sigma2, k_=None):
    """For details, see here.

    Parameters
    ----------
        sigma2 : array, shape (n_,n_)
        k_ : int, optional

    Returns
    -------
        e : array, shape (n_,k_)
        lambda2 : array, shape (k_,)

    """
    n_ = sigma2.shape[0]
    if k_ is None:
        k_ = n_
    lambda2, e = sp.linalg.eigh(sigma2, eigvals=(n_-k_, n_-1))
    lambda2 = lambda2[::-1]
    e = e[:, ::-1]

    # Enforce a sign convention on the coefficients
    # the largest element in each eigenvector will have a positive sign
    ind = np.argmax(abs(e), axis=0)
    ind = np.diag(e[ind, :]) < 0
    e[:, ind] = -e[:, ind]

    return e, lambda2
