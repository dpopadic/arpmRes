#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from arpym.statistics.cop_marg_sep import cop_marg_sep


def schweizer_wolff(x, p=None):
    """For details, see here.

    Parameters
    ----------
        x : array, shape (j_, 2)
        p : array, shape (j_, )

    Returns
    -------
        sw : scalar

    """

    j_ = x.shape[0]  # number of scenarios

    if p is None:
        p = np.ones(j_) / j_

    # Step 1: Compute grades scenarios
    u, _, _ = cop_marg_sep(x, p)

    # Step 2: Compute the joint scenario-probability cdf of grades
    cdf_u = np.array([np.sum(p * (u[:, 0] <= i / j_) * (u[:, 1] <= k / j_))
                      for i in range(j_) for k in range(j_)]).reshape((j_, j_))

    # Step 3: Approximate Schweizer-Wolff measure
    sw_abs = np.array([np.abs(cdf_u[i, k] - (i * k) / j_ ** 2)
                       for i in range(j_) for k in range(j_)]).reshape((j_, j_))
    sw = 12 / j_ ** 2 * np.sum(sw_abs)

    return sw
