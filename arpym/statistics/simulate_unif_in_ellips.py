# -*- coding: utf-8 -*-

import numpy as np

from arpym.statistics.simulate_normal import simulate_normal
from arpym.tools.transpose_square_root import transpose_square_root


def simulate_unif_in_ellips(mu, sigma2, j_):
    """For details, see here.

    Parameters
    ----------
        mu : array, shape (n_,)
        sigma2 : array, shape (n_,n_)
        j_ : int

    Returns
    -------
        x : array, shape(j_,n_)
        r : array, shape (j_,)
        y : array, shape (j,n_)

    """

    n_ = len(mu)

    # Step 1. Riccati root
    sigma = transpose_square_root(sigma2)

    # Step 2. Radial scenarios
    r = (np.random.rand(j_, 1))**(1/n_)

    # Step 3. Normal scenarios
    n = simulate_normal(np.zeros(n_), np.eye(n_), j_).reshape(-1, n_)

    # Step 4. Uniform component
    normalizers = np.linalg.norm(n, axis=1)
    y = n/normalizers[:, None]

    # Step 5. Output
    x = mu + r * y @ sigma

    return x, r, y
