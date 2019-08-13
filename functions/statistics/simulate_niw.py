# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import invwishart


def simulate_niw(mu, t, sigma2, nu, j_):
    """For details, see here.

    Parameters
    ----------
        mu : array, shape (n_,)
        t : float
        sigma2 : array, shape (n_, n_)
        nu : float
        j_ : int

    Returns
    -------
        m : array, shape (j_, n_)
        s2 : array, shape (j_, n_, n_)

    """

    if np.shape(mu) is ():
        n_ = 1
    else:
        n_ = np.shape(mu)[0]

    # Step 1: Generate inverse Wishart random matrices

    s2 = invwishart.rvs(nu, nu * sigma2, j_)

    if n_ == 1:
        # Step 2: If n_ = 1, then compute the random numbers via definition
        m = mu + np.sqrt(s2 / t) * np.random.randn(j_)
    else:
        # Step 3: If n_ > 1, then use Cholesky decomposition
        l = np.linalg.cholesky(s2)
        x = np.random.randn(j_, n_)
        m = np.zeros((j_, n_))

        # multiply each x[j,:] by the lower triangular matrix l[j,:,:]
        for i in range(n_):
            for j in range(i + 1):
                m[:, i] += l[:, i, j] * x[:, j]

        m = mu + m / np.sqrt(t)

    return m, s2
