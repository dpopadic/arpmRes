# -*- coding: utf-8 -*-
import numpy as np

from arpym.statistics.simulate_normal import simulate_normal


def simulate_bm(x_tnow, delta_t_m, mu, sig2, j_):
    """For details, see here.

    Parameters
    ----------
        x_t_now : array, shape(n_, )
        delta_t_m : array, shape(m_, )
        mu : array, shape(n_, )
        sig2 : array, shape(n_,n_)
        j_ : int

    Returns
    -------
        x_tnow_thor : array, shape(j_, m_ + 1, n_)

    """
    x_tnow, delta_t_m, mu, sig2 = \
        np.atleast_1d(x_tnow), np.atleast_1d(delta_t_m), np.atleast_1d(mu), np.atleast_2d(sig2)

    n_ = x_tnow.shape[0]
    m_ = delta_t_m.shape[0]

    x_tnow_thor = np.zeros((j_, m_ + 1, n_))

    # Step 1: Compute invariants

    x_tnow_thor[:, 0, :] = x_tnow

    for m in range(m_):
        epsi_delta_t_m = simulate_normal(mu * delta_t_m[m], sig2 * delta_t_m[m], j_)

        # Step 2: Compute projected path

        x_tnow_thor[:, m + 1, :] = x_tnow_thor[:, m, :] + epsi_delta_t_m.reshape(j_, n_)

    return x_tnow_thor
