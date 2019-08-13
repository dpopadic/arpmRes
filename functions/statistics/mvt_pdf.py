# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp


def mvt_pdf(x, mu, sig2, nu):
    """For details, see here.

    Parameters
    ----------
        x : array, shape (k_, n_)
        mu : array, shape (n_,)
        sig2 : array, shape (n_, n_)
        nu : int
    Returns
    -------
        f : array, shape (k_,)

    """

    x = np.atleast_2d(x)
    n_ = sig2.shape[0]
    f = np.array([sp.special.gamma((nu+n_)/2) /
                  (sp.special.gamma(nu/2) * (nu * np.pi)**(n_/2) *
                  np.sqrt(np.linalg.det(sig2))) *
                  (1+(x_k-mu).T@np.linalg.solve(sig2, x_k-mu)/nu)**(-(n_+nu)/2)
                 for x_k in x])

    return np.squeeze(f)
