# -*- coding: utf-8 -*-"

import numpy as np
import scipy.stats as stats
from bisect import bisect_left


def quantile_sp(c_, x, p=None, method=None, h=None):
    """For details, see here:

    Parameters
    ----------
        c_ : scalar, array, shape(k_,)
        x : array, shape (j_,)
        p : array, shape (j_,), optional
        method: string, optional
        h: scalar, optional

    Returns
    -------
        q : array, shape (k_,)

    """

    c_ = np.atleast_1d(c_)
    j_ = x.shape[0]
    k_ = c_.shape[0]

    # Step 1: Sorted scenarios-probabilities

    if p is None:
        p = np.ones(j_) / j_  # equal probabilities as default value

    sort_x = np.argsort(x)
    x_sort = x[sort_x]
    p_sort = p[sort_x]

    # Step 2: Cumulative sums of sorted probabilities

    u_sort = np.zeros(j_ + 1)
    for j in range(1, j_ + 1):
        u_sort[j] = np.sum(p_sort[:j])

    # Step 3: Output quantile

    q = np.zeros(k_)

    qindx = [0]*k_
    for k in range(k_):
        qindx[k] = bisect_left(u_sort, c_[k])-1
        if qindx[k] == -1:
            qindx[k] = 0
        if qindx[k] >= x.shape[0]:
            qindx[k] = x.shape[0]-1

    if method == "linear_interp":
        x_0 = x_sort[0]-(x_sort[1]-x_sort[0])*u_sort[1]/(u_sort[2]-u_sort[1])
        x_sort = np.append(x_0, x_sort)
        for k in range(k_):
            q[k] = x_sort[qindx[k]] + \
                       (x_sort[qindx[k]+1] - x_sort[qindx[k]]) *\
                       (c_[k]-u_sort[qindx[k]]) /\
                       (u_sort[qindx[k]+1]-u_sort[qindx[k]])

    elif method == "kernel_smoothing":
        if h is None:
            h = 0.25*(j_**(-0.2))

        for k in range(k_):
            w = np.diff(stats.norm.cdf(u_sort, c_[k], h))
            w = w / np.sum(w)
            q[k] = x_sort @ w

    else:
        q = x_sort[qindx]

    return np.squeeze(q)
