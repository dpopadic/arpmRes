#!/usr/bin/env python3

import numpy as np

from arpym.statistics.project_trans_matrix import project_trans_matrix


def simulate_markov_chain_univ(x_tnow, p, delta_t_m, j_=1000):
    """For details, see here.
    Parameters
    ----------
        x_tnow : scalar
        p : array, shape(c_, c_)
        delta_t_m : array, shape(m_, )
        j_ : int
    Returns
    -------
        x_tnow_thor : array, shape(j_, m_ + 1)
    """

    m_ = delta_t_m.shape[0]

    x_tnow_thor = np.zeros((j_, m_ + 1))
    x_tnow_thor[:, 0] = x_tnow

    for m in np.arange(m_):

        # 1. Step 1: Compute invariants

        epsi = np.random.uniform(0, 1, j_)

        # 2. Step 2: Compute projected path

        # transition matrix
        p_dt = project_trans_matrix(p, delta_t_m[m])
        for j in np.arange(j_):
            # thresholds for quantile
            f = np.r_[0, np.cumsum(p_dt[int(x_tnow_thor[j, m]), :])]

            # compute state
            x_tnow_thor[j, m + 1] = np.sum(f <= epsi[j]) - 1

    return x_tnow_thor
