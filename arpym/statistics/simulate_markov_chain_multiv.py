#!/usr/bin/env python3

import numpy as np
import scipy.stats as st

from arpym.statistics.simulate_t import simulate_t


def simulate_markov_chain_multiv(x_tnow, p, m_, *, rho2=None, nu=None, j_=1000):
    """For details, see here.

    Parameters
    ----------
        x_tnow : array, shape(d_, )
        p : array, shape(s_, s_)
        deltat_m : int
        rho2 : array, shape(d_, d_)
        nu : int
        j_ : int

    Returns
    -------
        x_tnow_thor : array, shape(j_, m_ + 1, d_)

    """

    d_ = x_tnow.shape[0]

    if rho2 is None:
        # uncorrelated marginal transitions
        rho2 = np.eye(d_)

    if nu is None:
        nu = 10 ** 9  # normal copula

    x_tnow_thor = np.zeros((j_, m_ + 1, d_))
    x_tnow_thor[:, 0, :] = x_tnow

    for m in np.arange(m_):

        # Step 1: Generate copula scenarios and compute their grades

        # scenarios from a t copula
        u = simulate_t(np.zeros(d_), rho2, nu, j_).reshape((j_, d_))

        # grades
        epsi = st.t.cdf(u, nu)

        # Step 2: Compute projected path

        for j in np.arange(j_):
            for d in np.arange(d_):
                # thresholds for quantile
                f = np.r_[0, np.cumsum(p[int(x_tnow_thor[j, m, d]), :])]

                # state
                x_tnow_thor[j, m + 1, d] = np.sum(f <= epsi[j, d])-1

    return x_tnow_thor
