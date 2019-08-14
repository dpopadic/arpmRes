#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


def histogram2d_sp(x, p=None, k_=None, xi=None):
    """For details, see here.

    Parameters
    ----------
        x : array, shape(j_, 2)
        p : array, shape(j_ )
        k_ : int, optional
        xi : array, shape (k_, 2), optional

    Returns
    -------
        f : array, shape (k_, k_)
        xi_1 : array, shape (k_, )
        xi_2 : array, shape(k_, )

    """

    j_ = x.shape[0]

    if p is None:
        # uniform probabilities
        p = np.ones(j_) / j_

    if k_ is None and xi is None:
        # Sturges formula
        k_ = 10.0*np.log(j_)
    if xi is not None:
        k_ = xi.shape[0]

    k_ = int(k_)

    min_x1 = np.min(x[:, 0])
    min_x2 = np.min(x[:, 1])

    # Step 1: Compute bin width

    if xi is None:
        h_1 = (np.max(x[:, 0]) - min_x1)/k_
        h_2 = (np.max(x[:, 1]) - min_x2)/k_
    else:
        h_1 = xi[1, 0] - xi[0, 0]
        h_2 = xi[1, 1] - xi[0, 1]

    # Step 2: Compute bin centroids

    if xi is None:
        xi_1 = np.zeros(k_)
        xi_2 = np.zeros(k_)
        for k in range(k_):
            xi_1[k] = min_x1 + (k+1-0.5)*h_1
            xi_2[k] = min_x2 + (k+1-0.5)*h_2
    else:
        xi_1 = xi[:, 0]
        xi_2 = xi[:, 1]

    # Step 3: Compute the normalized histogram heights

    f = np.zeros((k_, k_))

    for k_1 in range(k_):
            for k_2 in range(k_):
                # Take edge cases into account
                if k_1 > 0 and k_2 > 0:
                    ind = ((x[:, 0] > xi_1[k_1]-h_1/2) & (x[:, 0] <= xi_1[k_1]+h_1/2) &
                           (x[:, 1] > xi_2[k_2]-h_2/2) & (x[:, 1] <= xi_2[k_2]+h_2/2))
                elif k_1 > 0 and k_2 == 0:
                    ind = ((x[:, 0] > xi_1[k_1]-h_1/2) & (x[:, 0] <= xi_1[k_1]+h_1/2) &
                           (x[:, 1] >= min_x2) & (x[:, 1] <= xi_2[k_2]+h_2/2))
                elif k_1 == 0 and k_2 > 0:
                    ind = ((x[:, 0] >= min_x1) & (x[:, 0] <= xi_1[k_1]+h_1/2) &
                           (x[:, 1] > xi_2[k_2]-h_2/2) & (x[:, 1] <= xi_2[k_2]+h_2/2))
                else:
                    ind = ((x[:, 0] >= min_x1) & (x[:, 0] <= xi_1[k_1]+h_1/2) &
                           (x[:, 1] >= min_x2) & (x[:, 1] <= xi_2[k_2]+h_2/2))

                f[k_1, k_2] = np.sum(p[ind])/(h_1*h_2)

    return f, xi_1, xi_2
