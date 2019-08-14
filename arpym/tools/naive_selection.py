# -*- coding: utf-8 -*-

import numpy as np


def naive_selection(g, n_, k_=None):
    """For details, see here.

    Parameters
    ----------
        g : function
        n_ : int
        k_ : int, optional

    Returns
    -------
        s_star_naive : list, shape(k_, 1:k_)

    """
    if k_ is None:
        k_ = n_

    # Step 1: evaluate the performance function
    g_n = np.array([g(np.array([n])) for n in range(1, n_+1)])

    # Step 2: sort based on the performance function
    n_sort = np.argsort(-g_n) + 1

    # Step 3: select the first k_ sets
    return [n_sort[:k] for k in range(1, k_+1)]
