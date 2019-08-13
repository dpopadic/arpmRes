# -*- coding: utf-8 -*-

import numpy as np
from arpym.statistics.simulate_normal import simulate_normal


def simulate_wishart(nu, sig2, j_):
    """For details, see here.

    Parameters
    ----------
        nu: int
        sig2_ : array, shape (n_, n_)
        j_ : int

    Returns
    -------
        w2 : array, shape (j_, n_, n_) if j_>1 or (n_, n_) for j_=1

    """

    n_ = sig2.shape[0]

    # generate normal Monte Carlo simulations
    x = simulate_normal(np.zeros(n_), sig2, nu*j_).reshape((j_, nu, n_))

    # generate Monte Carlo simulations of Wishart random matrix
    w2 = np.transpose(x, (0, 2, 1))@x

    return np.squeeze(w2)
