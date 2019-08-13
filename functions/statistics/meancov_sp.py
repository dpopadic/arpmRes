# -*- coding: utf-8 -*-

import numpy as np


def meancov_sp(x, p=None):
    """For details, see here.

    Parameters
    ----------
        x : array, shape (j_,n_) if n_>1 or (j_,) if n_=1
        p : array, shape (j_,)

    Returns
    -------
        e_x : array, shape (n_,)
        cv_x : array, shape (n_,n_ )

    """

    if p is None:
        j_ = x.shape[0]
        p = np.ones(j_) / j_  # equal probabilities as default value

    e_x = p @ x
    cv_x = ((x-e_x).T*p) @ (x-e_x)
    return e_x, cv_x
