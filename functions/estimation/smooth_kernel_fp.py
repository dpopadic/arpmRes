# -*- coding: utf-8 -*-
import numpy as np


def smooth_kernel_fp(z, z_star, h, gamma=2):
    """For details, see here.

    Parameters
    ----------
        z : array, shape (t_, )
        z_star : scalar
        h : scalar
        gamma : scalar

    Returns
    -------
        p : array, shape(t_, )

    """

    # compute probabilities
    p = np.exp(-(np.abs(z-z_star)/h)**gamma)
    # rescale
    p = p / np.sum(p)
    return np.squeeze(p)
