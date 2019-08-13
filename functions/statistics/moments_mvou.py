# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp


def moments_mvou(x_tnow, deltat_m, theta, mu, sig2):
    """For details, see here.

    Parameters
    ----------
        x_tnow : array, shape(n_, )
        deltat_m : array, shape(m_, )
        theta : array, shape(n_, n_)
        mu : array, shape(n_, )
        sig2 : array, shape(n_, n_)

    Returns
    -------
        mu_dt_m : array, shape(m_, n_)
        mu_deltat_m : array, shape(m_, n_) AC: mu_deltat_m
        sig2_deltat_m : array, shape(m_, n_, n_)

    """

    if len(x_tnow.shape) != 1:
        x_tnow = x_tnow.reshape(-1, 1).copy()
    if len(mu.shape) != 1:
        mu = mu.reshape(-1, 1).copy()

    n_ = x_tnow.shape[0]

    if isinstance(deltat_m, float) or isinstance(deltat_m, np.int64):
        m_ = 1
        deltat_m = np.array([deltat_m])
    else:
        m_ = len(deltat_m)

    mu_dt_m = np.zeros((m_, n_))
    mu_deltat_m = np.zeros((m_, n_))
    sig2_deltat_m = np.zeros((m_, n_, n_))

    for m, tm in np.ndenumerate(deltat_m):

        # Step 1: compute drift of shocks
        mu_dt_m[[m], :] = (np.eye(n_) - sp.linalg.expm(-theta * tm)) \
                                @ (np.linalg.solve(theta, mu))

        # Step 2: compute drift of process
        mu_deltat_m[[m], :] = sp.linalg.expm(-theta * tm) @ x_tnow + \
            mu_dt_m[[m], :]

        # Step 3: compute covariance of process
        th_sum_th = sp.linalg.kron(theta, np.eye(n_)) + \
            sp.linalg.kron(np.eye(n_), theta)
        vecsig2 = np.reshape(sig2, (n_ ** 2, 1), 'F')
        vecsig2_m = np.linalg.solve(th_sum_th, (np.eye(n_ ** 2) -
                                    sp.linalg.expm(-th_sum_th * tm))) @ vecsig2
        sig2_m = np.reshape(vecsig2_m, (n_, n_), 'F')

        # grant numerical symmetry
        sig2_deltat_m[[m], :, :] = (sig2_m + sig2_m.T) / 2

    # resize
    if n_ != 1:
        mu_dt_m = mu_dt_m.squeeze()
        mu_deltat_m = mu_deltat_m.squeeze()
    sig2_deltat_m = np.atleast_2d(sig2_deltat_m.squeeze())

    return mu_dt_m, mu_deltat_m, sig2_deltat_m
