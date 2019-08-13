# -*- coding: utf-8 -*-
import numpy as np


def histogram_sp(x, *, p=None, k_=None, xi=None):
    """For details, see here.

    Parameters
    ----------
        x : array, shape(j_,)
        p : array, shape(j_), optional
        k_ : scalar, optional
        xi : array, shape (k_, ), optional

    Returns
    -------
        f : array, shape (n_bins,)
        xi : array, shape (n_bins,)

    """

    j_ = x.shape[0]

    if p is None:
        # uniform probabilities
        p = np.ones(j_) / j_

    if k_ is None and xi is None:
        # Sturges formula
        k_ = np.ceil(np.log(j_)) + 1
    if xi is not None:
        k_ = xi.shape[0]

    k_ = int(k_)

    minx = np.min(x)

    # Step 1: Compute bin width

    if xi is None:
        h = (np.max(x) - minx) / k_
    else:
        h = xi[1] - xi[0]

    # Step 2: Compute bin centroids

    if xi is None:
        xi = np.zeros(k_)
        for k in range(k_):
            xi[k] = minx + (k + 1 - 0.5) * h

    # Step 3: Compute the normalized histogram heights

    f = np.zeros(k_)

    f[0] = np.sum(p[(x >= minx) & (x <= xi[0] + h / 2)]) / h

    for k in range(1, k_):
        ind = (x > xi[k] - h / 2) & (x <= xi[k] + h / 2)
        f[k] = np.sum(p[ind]) / h

    return np.squeeze(f), np.squeeze(xi)
