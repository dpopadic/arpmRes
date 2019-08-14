# -*- coding: utf-8 -*-
import numpy as np

from scipy.linalg import lu


def rel_entropy_normal(mu_x_updated, sigma2_x_updated,
                       mu_x_base, sigma2_x_base):
    """For details, see here.

    Parameters
    ----------
        mu_x_updated : array, shape (n_,)
        sigma2_x_updated : array, shape (n_, n_)
        mu_x_base : array, shape (n_,)
        sigma2_x_base : array, shape (n_, n_)

    Returns
    -------
        relative_entropy : scalar

    """

    # Fast logarithm-determinant computation
    if (np.ndim(mu_x_updated) == 0) or (np.ndim(mu_x_base) == 0):
        mu_x_updated, sigma2_x_updated, mu_x_base, sigma2_x_base =\
            mu_x_updated.reshape(-1).copy(),\
            sigma2_x_updated.reshape(1, 1).copy(),\
            mu_x_base.reshape(-1).copy(),\
            sigma2_x_base.reshape(1, 1).copy()

    def logdet(a):
        p, l, u = lu(a)
        v = np.log(abs(np.prod(np.r_[np.diag(l), np.diag(u)])))
        return v

    n_ = sigma2_x_base.shape[0]

    inv_sigma2_x_base = np.linalg.solve(sigma2_x_base, np.eye(n_))
    sigma2_x_updated_inv_sigma2_x_base = sigma2_x_updated @\
        inv_sigma2_x_base
    mu_x_updated = mu_x_updated.reshape(-1, 1)
    mu_x_base = mu_x_base.reshape(-1, 1)
    mu_diff = mu_x_updated - mu_x_base
    relative_entropy = 0.5*(np.trace(sigma2_x_updated_inv_sigma2_x_base) -
                            logdet(sigma2_x_updated_inv_sigma2_x_base) +
                            mu_diff.T @ inv_sigma2_x_base @ mu_diff -
                            n_)
    return np.squeeze(relative_entropy)
