# -*- coding: utf-8 -*-

import numpy as np


def cov_2_corr(s2):
    """For details, see here.

    Parameters
    ----------
        s2 : array, shape (n_, n_)

    Returns
    -------
        c2 : array, shape (n_, n_)
        s_vol : array, shape (n_,)

    """
    # compute standard deviations
    s_vol = np.sqrt(np.diag(s2))

    diag_inv_svol = np.diag(1/s_vol)
    # compute correlation matrix
    c2 = diag_inv_svol@s2@diag_inv_svol
    return c2, s_vol
