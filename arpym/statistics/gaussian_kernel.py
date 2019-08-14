# -*- coding: utf-8 -*-

import numpy as np


def gaussian_kernel(h2, y, x):
    """For details, see here.

    Parameters
    ----------
        h2: scalar
        y : array, shape (n_, )
        x : array, shape (n_, )

    Returns
    -------
        delta_h2_y_x : scalar

    """
    if np.ndim(x) == 0:
        x = np.atleast_1d(x)
    if np.ndim(y) == 0:
        y = np.atleast_1d(y)
    n_ = y.shape[0]

    delta_h2_y_x = 1/((2*np.pi*h2)**(n_/2))*np.exp(-(x-y).T@(x-y)/(2*h2))

    return np.squeeze(delta_h2_y_x)
