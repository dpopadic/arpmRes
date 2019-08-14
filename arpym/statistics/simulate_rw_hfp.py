# -*- coding: utf-8 -*-

import numpy as np

from arpym.statistics.bootstrap_hfp import bootstrap_hfp


def simulate_rw_hfp(x_t_now, epsi, p=None, j_=1, m_=1):
    """For details, see here.

    Parameters
    ----------
        x_t_now : array, shape(i_, )
        epsi : array, shape(t_, i_)
        p : array, shape(t_, )
        j_ : int
        m_ : int

    Returns
    -------
        x_tnow_thor : array, shape(j_, m_ + 1, i_)

    """

    if p is None:
        t_ = epsi.shape[0]
        p = np.ones(t_) / t_

    i_ = x_t_now.shape[0]
    x_tnow_thor = np.zeros((j_, m_ + 1, i_))
    x_tnow_thor[:, 0, :] = x_t_now

    for m in range(1, m_ + 1):
        # Step 1: Compute invariants via bootstraping

        epsi_j = bootstrap_hfp(epsi, p, j_)

        # Step 2: Compute projected path

        x_tnow_thor[:, m, :] = x_tnow_thor[:, m - 1, :] + epsi_j

    return x_tnow_thor
