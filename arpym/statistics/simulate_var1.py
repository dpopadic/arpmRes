# -*- coding: utf-8 -*-
import numpy as np

from arpym.statistics.simulate_t import simulate_t


def simulate_var1(x_tnow, b, mu, sigma2, m_, *, j_=1000, nu=10**9,
                  init_value=True):
    """For details, see here.

    Parameters
    ----------
        x_tnow : array, shape(n_, )
        b : array, shape(n_,n_)
        mu : array, shape(n_, )
        sigma2 :  array, shape(n_,n_)
        m_ : int
        nu: int
        j_ : int
        init_value : boolean

    Returns
    -------
        x_tnow_thor : array, shape(j_, m_+1, n_)

    """

    n_ = np.shape(sigma2)[0]

    # Step 1: Monte Carlo scenarios of projected paths of the risk drivers

    x_tnow_thor = np.zeros((j_, m_, n_))

    for m in range(0, m_):
        epsi = simulate_t(mu, sigma2, nu, j_).reshape((j_, -1))
        if m > 0:
            x_prec = x_tnow_thor[:, m-1, :]
        else:
            x_prec = np.tile(x_tnow, (j_, 1))

        x_tnow_thor[:, m, :] = x_prec @ b.T + epsi

    # Step 2: Include the initial value as starting node, if selected

    if init_value:
        x_tnow = np.tile(x_tnow, (j_, 1))
        x_tnow = np.expand_dims(x_tnow, axis=1)
        x_tnow_thor = np.concatenate((x_tnow, x_tnow_thor), axis=1)

    return x_tnow_thor
