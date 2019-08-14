# -*- coding: utf-8 -*-

import numpy as np


def meancov_wishart(nu, sig2):
    """For details, see here.

    Parameters
    ----------
        nu : int
        sig2 : array, shape (n_, n_)

    Returns
    -------
        e_w2 : array, shape (n_, n_)
        cv_w2 : array, shape (n_*n_, n_*n_)

    """

    n_ = sig2.shape[0]

    matrices = np.array([np.kron(np.kron(np.eye(n_)[k].reshape(-1, 1),
                                         np.eye(n_)),
                                 np.eye(n_)[k].reshape(-1, 1).T)
                        for k in range(n_)])
    # commutation matrix
    k_nn = np.sum(matrices, axis=0)

    # expectation of the Wishart distribution
    e_w2 = nu*sig2

    # covariance of the Wishart distribution
    cv_w2 = nu*(np.eye(n_**2)+k_nn)@np.kron(sig2, sig2)

    return e_w2, cv_w2
