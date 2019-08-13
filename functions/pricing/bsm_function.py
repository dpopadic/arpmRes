# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import norm


def bsm_function(s, r, sigma, m, tau):
    """For details, see here.

    Parameters
    ----------
        s : scalar
        r : scalar
        sigma : scalar
        m : scalar
        tau : scalar
        put : boolean

    Return
    ------
        c_bs : scalar


    """
    d1 = (m+(r+(sigma**2)/2)*np.sqrt(tau))/sigma
    d2 = d1-sigma*np.sqrt(tau)
    c_bs = s*(norm.cdf(d1) - np.exp(-(m*np.sqrt(tau)+r*tau))*norm.cdf(d2))

    return c_bs
