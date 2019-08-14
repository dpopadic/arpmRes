# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp


def solve_riccati(sigma2, phi2=None):
    """For details, see here.

    Parameters
    ----------
        sigma2 : array, shape (n_,n_)
        phi2 : array, shape (n_,n_), optional

    Returns
    -------
        b : array, shape (n_,n_)

    """
    n_ = sigma2.shape[0]

    if phi2 is None:
        phi2 = np.identity(n_)

    # Step 1. Hamiltonian matrix (block matrix)
    h = np.r_[np.r_['-1', np.zeros((n_, n_)), -phi2],
              np.r_['-1', -sigma2, np.zeros((n_, n_))]]

    # Step 2. Schur decomposition
    t, u, _ = sp.linalg.schur(h, output='real', sort=lambda x: x.real < 0)

    # Step 3. Four n_ x n_ partitions
    u_11 = u[:n_, :n_]
    u_21 = u[n_:, :n_]

    # Step 4. Riccati root
    b = u_21 @ np.linalg.inv(u_11)

    return b
