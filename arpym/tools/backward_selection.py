# -*- coding: utf-8 -*-

import numpy as np


def backward_selection(g, n_, k_=None):
    """For details, see here.

    Parameters
    ----------
        g : function
        n_ : int
        k_ : int, optional

    Returns
    -------
        s_star_bwd : list, shape(k_, 1:k_)

    """
    if k_ is None:
        k_ = n_

    # Step 0: Initialize
    s_star_backward_n_ = np.arange(1, n_+1)
    s_star_backward = []
    s_star_backward.append(s_star_backward_n_)
    for k in range(n_, 1, -1):

        # Step 1: Determine the worst element
        n_star = np.argmax([g(np.delete(s_star_backward[-1], i))
                            for i in range(s_star_backward[-1].shape[0])])

        # Step 2: Drop the worst element
        s_star_backward.append(np.delete(s_star_backward[-1], n_star))
    return list(reversed(s_star_backward[-k_:]))
