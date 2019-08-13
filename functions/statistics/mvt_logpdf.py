# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp


def mvt_logpdf(x, mu, sigma2, nu):
    """For details, see here.

    Parameters
    ----------
        x : array, shape (k_, n_)
        mu : array, shape (n_,)
        sigma2 : array, shape (n_, n_)
        nu: float, optional

    Returns
    -------
        lf : array, shape (k_,)

    """
    if np.shape(sigma2) is ():
        # univaraite student t
        lf = sp.stats.t.logpdf(x, nu, mu, sigma2)
    else:
        # multivariate student t
        n_ = sigma2.shape[0]
        d2 = np.sum((x - mu).T * np.linalg.solve(sigma2, (x - mu).T), axis=0)
        lf = -((nu + n_) / 2.) * np.log(1. + d2 / nu) + \
            sp.special.gammaln((nu + n_) / 2.) - \
            sp.special.gammaln(nu / 2.) - \
            (n_ / 2.) * np.log(nu * np.pi) - \
            0.5 * np.linalg.slogdet(sigma2)[1]

    return lf
