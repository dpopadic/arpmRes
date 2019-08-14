# -*- coding: utf-8 -*-

import numpy as np

from arpym.tools.solve_riccati import solve_riccati
from arpym.tools.cpca_cov import cpca_cov
from arpym.tools.pca_cov import pca_cov
from arpym.tools.gram_schmidt import gram_schmidt


def transpose_square_root(sigma2, method='Riccati', d=None):
    """For details, see here.

    Parameters
    ----------
        sigma2 : array, shape (n_,n_)
        method : string, optional
        d : array, shape (k_,n_), optional

    Returns
    -------
        s : array, shape (n_,n_)

    """
    if np.ndim(sigma2) < 2:
        return np.squeeze(np.sqrt(sigma2))

    n_ = sigma2.shape[0]

    if method == 'CPCA' and d is None:
        method = 'PCA'

    # Step 1: Riccati root
    if method == 'Riccati':
        s = solve_riccati(sigma2)

    # Step 2: Conditional principal components
    elif method == 'CPCA':
        e_d, lambda2_d = cpca_cov(sigma2, d)
        s = e_d * np.sqrt(lambda2_d)

    # Step 3: Principal components
    elif method == 'PCA':
        e, lambda2 = pca_cov(sigma2)
        s = e * np.sqrt(lambda2)

    # Step 4: Gram-Schmidt
    elif method == 'Gram-Schmidt':
        g = gram_schmidt(sigma2)
        s = np.linalg.inv(g).T

    # Step 5: Cholesky
    elif method == 'Cholesky':
        s = np.linalg.cholesky(sigma2)

    return s
