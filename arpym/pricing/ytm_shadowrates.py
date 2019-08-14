#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def ytm_shadowrates(y, eta=0.013):
    """For details, see here.

    Parameters
    ----------
        y : array, shape(t_,d_) if d_>1 or (t_,) for d_=1
        eta : scalar


    Returns
    -------
        x : array, shape(t_,d_) if d_>1 or (t_,) for d_=1

    """

    x = np.zeros((y.shape))

    # Compute inverse call transformation

    x[y >= eta] = y[y >= eta]
    x[y < eta] = eta * (np.log(y[y < eta]/eta)+1)

    return np.squeeze(x)
