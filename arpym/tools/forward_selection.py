# -*- coding: utf-8 -*-

import numpy as np


def forward_selection(g, n_, k_=None):
    """For details, see here.

    Parameters
    ----------
        g : function
        n_ : int
        k_ : int, optional

    Returns
    -------
        s_star_fwd : list, shape(k_, 1:k_)

    """
    if k_ is None:
        k_ = n_

    # Step 0: Initialize
    j = np.arange(1, n_ + 1)
    s_star_forward = np.array([], dtype='int')
    for k in range(k_):
        # Step 1: Determine the best element in j
        n_star = np.argmax([g(np.append(s_star_forward, n))
                            for n in j])

        # Step 2: Append the best element in j to the selection and remove it from j
        s_star_forward = np.append(s_star_forward, j[n_star])
        j = np.delete(j, n_star)

    return [s_star_forward[:k] for k in range(1, k_ + 1)]
