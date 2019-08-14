# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from arpym.tools.transpose_square_root import transpose_square_root


def multi_r2(s2_u, s2_x, sigma2=None):
    """For details, see here.

    Parameters
    ----------
        s2_u : array, shape(n_, n_)
        s2_x : array, shape(n_, n_)
        s2: array, shape(n_, n_)

    Returns
    -------
        r2 : float

    """

    if sigma2 is None:
        r2 = 1.0 - np.trace(s2_u) / np.trace(s2_x)  # r-squared
    else:
        sigma_cholesky = transpose_square_root(sigma2, method='Cholesky') # Cholesky root
        ss_res = np.trace(sp.linalg.solve_triangular(
                sigma_cholesky, sp.linalg.solve_triangular(sigma_cholesky, s2_u, lower=True).T).T)
        ss_tot = np.trace(sp.linalg.solve_triangular(
                sigma_cholesky, sp.linalg.solve_triangular(sigma_cholesky, s2_x, lower=True).T).T)
        r2 = 1.0 - ss_res / ss_tot  # r-squared

    return r2
