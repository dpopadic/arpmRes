# -*- coding: utf-8 -*-

import numpy as np


def mahalanobis_dist(x, m, s2):
    """For details, see here.

    Parameters
    ----------
        x : array, shape (t_, n_)
        m : array, shape (n_,)
        s2 : array, shape (n_, n_)

    Returns
    -------
        d : array, shape (t_,)

    """
    if np.ndim(x) > 1:
        d = np.sqrt(np.sum((x-m).T * np.linalg.solve(s2, (x-m).T), axis=0))
    else:
        d = np.sqrt((x-m) / s2 * (x-m))
    return np.squeeze(d)
