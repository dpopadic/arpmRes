# -*- coding: utf-8 -*-

import numpy as np


def moments_logn(b, mu, sigma2, a):
    """For details, see here.

    Parameters
    ----------
        b : array, shape (n_,)
        mu : array, shape (n_,)
        sigma2 : array, shape (n_, n_)
        a : array, shape(n_,)

    Returns
    -------
        mu_Y : scalar
        sd_Y : scalar
        sk_Y : scalar

    """

    n_ = len(mu)
    if a is None:
        a = np.zeros((n_,))

    # Expectation
    mulgn = np.exp(mu + 1/2*np.diag(sigma2))
    mu_Y = np.array([b.T@mulgn-b.T@a])

    # Initialize covariances and third central moments
    Cov = np.zeros((n_, n_))
    cent3rd = np.zeros((n_**3, 1))
    i = 0

    for n in range(n_):
        for m in range(n_):
            # Covariances
            Cov[n, m] = np.exp(mu[m]+mu[n] + 1/2 * (sigma2[n, n] +
                               sigma2[m, m])) * (np.exp(sigma2[m, n])-1)
            noncent2nd_nm = np.exp(mu[m]+mu[n] +
                                   1/2*(sigma2[n, n] + sigma2[m, m]) +
                                   sigma2[n, m])
        for l in range(n_):
            i = i+1
            # second non-central moments that enter in the third non-central
            # moments formulas
            noncent2nd_nl = np.exp(mu[n]+mu[l] +
                                   1/2*(sigma2[n, n]+sigma2[l, l]) +
                                   sigma2[n, l])
            noncent2nd_ml = np.exp(mu[m]+mu[l] +
                                   1/2*(sigma2[m, m]+sigma2[l, l]) +
                                   sigma2[m, l])
            # third non-central moments
            noncent3rd_nml = np.exp(mu[m]+mu[n]+mu[l] +
                                    1/2*(sigma2[n, n]+sigma2[m, m] +
                                         sigma2[l, l])+sigma2[n, l] +
                                    sigma2[n, m]+sigma2[l, m])
            cent3rd[i] = noncent3rd_nml+2*mulgn[l]*mulgn[m]*mulgn[n] - \
                noncent2nd_nm*mulgn[l] - noncent2nd_nl*mulgn[m] - \
                noncent2nd_ml*mulgn[n]

    sd_Y = np.sqrt(np.array([b.T@Cov@b]))  # standard deviation
    dummy = np.kron(np.kron(b, b), b)
    vec_h = dummy.flatten()
    sk_Y = (vec_h.T@cent3rd)/(sd_Y**3)  # skewness

    return mu_Y, sd_Y, sk_Y
