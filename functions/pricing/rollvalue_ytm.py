# -*- coding: utf-8 -*-

import numpy as np


def rollvalue_ytm(tau, v):
    """For details, see here.

    Parameters
    ----------
        tau : array, shape (d_,)
        v : array, shape (t_, d_)

    Returns
    -------
        y : array, shape (t_, d_)
        log_v : array, shape (t_, d_)

    """

    log_v = np.log(v)  # log-rolling values
    y = -(1 / tau) * log_v  # yields to maturity
    return y, log_v
