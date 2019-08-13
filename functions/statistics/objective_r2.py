# -*- coding: utf-8 -*-

import numpy as np

from arpym.statistics.multi_r2 import multi_r2


def objective_r2(s_k, s2_xz, n_, sigma2=None):
    """For details, see here.

    Parameters
    ----------
        s_k : array, shape(i_,)
        s2_xz : array, shape(n_+k_, n_+k_)
        n_ : int
        sigma2 : array, shape(n_, n_), optional

    Returns
    -------
        r2 : float

    """
    s_k = s_k - 1
    s2_x = s2_xz[:n_, :n_]
    s_xz = s2_xz[:n_, n_:][:, s_k]
    s2_z = s2_xz[n_:, n_:][s_k, :][:, s_k]
    s2_z_inverse = np.linalg.inv(s2_z)
    s2_u = s2_x - s_xz @ s2_z_inverse @ (s_xz).T

    r2 = multi_r2(s2_u, s2_x, sigma2)

    return r2
