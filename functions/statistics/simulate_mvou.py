# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp

from arpym.statistics.moments_mvou import moments_mvou
from arpym.statistics.simulate_normal import simulate_normal


def simulate_mvou(x_tnow, delta_t_m, theta, mu, sig2, j_=1000, init_value=True):

    """For details, see here.

    Parameters
    ----------
        x_tnow : array, shape(n_, )
        delta_t_m : array, shape(m_, )
        theta: array, shape(n_, n_)
        mu : array, shape(n_, )
        sig2 : array, shape(n_,n_)
        j_ : scalar
        init_value : boolean

    Returns
    -------
        x_tnow_thor : array, shape(j_, m_, n_)

    """

    if len(x_tnow.shape) != 1:
        x_tnow = x_tnow.reshape(-1, 1)
    if len(mu.shape) != 1:
        mu = mu.reshape(-1, 1)

    n_ = x_tnow.shape[0]
    m_ = delta_t_m.shape[0]

    # Step 1: Compute moments for each time interval

    x_tnow_thor = np.zeros((j_, m_, n_))

    for m in range(m_):

        # deltat_m-steps first and second moments
        mu_delta_t, _, sig2_delta_t = moments_mvou(np.zeros(n_), delta_t_m[m],
                                                   theta, mu, sig2)

        # Step 2: Monte Carlo scenarios of projected paths

        # compute the projected invariants
        epsi = simulate_normal(mu_delta_t, sig2_delta_t, j_).reshape(j_, -1)

        # compute risk drivers
        if m > 0:
            x_prec = x_tnow_thor[:, m-1, :]
        else:
            x_prec = np.tile(x_tnow, (j_, 1))

        x_tnow_thor[:, m, :] = x_prec @ \
            sp.linalg.expm(-theta * delta_t_m[m]).T + epsi

    # Step 3: Include the initial value as starting node, if selected
    if init_value:
        x_tnow = np.tile(x_tnow, (j_, 1))
        x_tnow = np.expand_dims(x_tnow, axis=1)
        x_tnow_thor = np.concatenate((x_tnow, x_tnow_thor), axis=1)

    return x_tnow_thor.squeeze()
