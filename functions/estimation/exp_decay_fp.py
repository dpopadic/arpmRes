# -*- coding: utf-8 -*-

import numpy as np


def exp_decay_fp(t_, tau_hl, t_star=None):
    """For details, see here.

    Parameters
    ----------
        t_ : scalar
        tau_hl : scalar
        t_star: scalar, optional
    Returns
    -------
        p array, shape(t_, )

    """

    if t_star is None:
        t_star = t_

    # compute probabilities
    p = np.exp(-(np.log(2) / tau_hl) * abs(t_star - np.arange(0, t_)))
    # rescale
    p = p / np.sum(p)
    return p
