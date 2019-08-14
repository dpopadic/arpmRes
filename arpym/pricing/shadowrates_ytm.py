# -*- coding: utf-8 -*-

import numpy as np


def shadowrates_ytm(x, eta=0.013):
    """For details, see here.

    Parameters
    ----------
        x : array, shape(t_, d_) if d_>1 or (t_,) for d_=1
        eta : scalar


    Returns
    -------
        y : array, shape(t_, d_) if d_>1 or (t_,) for d_=1

    """

    y = np.zeros((x.shape))

    y[x >= eta] = x[x >= eta]
    y[x < eta] = eta * np.exp(x[x < eta]/eta - 1)

    return np.squeeze(y)
