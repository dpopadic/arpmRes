# -*- coding: utf-8 -*-

import numpy as np


def meancov_inverse_wishart(nu, psi2):
    """For details, see here.

    Parameters
    ----------
        nu: int
        psi2 : array, shape (n_, n_)

    Returns
    -------
        e_sig2 : array, shape (n_, n_)
        cv_sig2 : array, shape (n_*n_, n_*n_)

    """

    n_ = psi2.shape[0]
    # expectation of the inverse-Wishart distribution
    e_sig2 = 1/(nu-n_-1)*psi2

    n_indices = [n for n in range(n_)]
    ind = [[m, n, p, q] for m in n_indices for n in n_indices
           for p in n_indices for q in n_indices]

    cv = np.zeros(n_**4)
    for i in range(len(ind)):
        m, n, p, q = ind[i][0], ind[i][1], ind[i][2], ind[i][3]

    # cross covariance
        cv_sig2_mnpq = (2 * psi2[m, n] * psi2[p, q] +
                        (nu - n_ - 1) * (psi2[m, p] * psi2[n, q] +
                        psi2[m, q] * psi2[n, p])) /\
                       ((nu-n_) * (nu-n_-1)**2 * (nu-n_-3))

        cv[i] = cv_sig2_mnpq

    # covariance of the inverse-Wishart distribution
    cv_sig2 = cv.reshape(n_**2, n_**2)

    return e_sig2, cv_sig2
