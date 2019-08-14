# -*- coding: utf-8 -*-
import numpy as np
from sklearn.covariance import graphical_lasso
from arpym.estimation.cov_2_corr import cov_2_corr


def markov_network(sigma2, k, lambda_vec, tol=10 ** -14, opt=False):
    """ For details, see here.

    Parameters
    ----------
        sigma2 : array, shape(n_, n_)
        k : scalar
        lambda_vec : array, shape(l_,)
        tol : scalar
        opt : bool

    Returns
    ----------
        sigma2_bar : array, shape(n_, n_, l_)
        c2_bar : array, shape(n_, n_, l_)
        phi2_bar : array, shape(n_, n_, l_)
        lambda_bar : scalar
        conv : scalar
        l_bar : scalar

    """

    lambda_vec = np.sort(lambda_vec)

    l_ = len(lambda_vec)

    c2_bar = np.zeros(sigma2.shape + (l_,))
    phi2_bar = np.zeros(sigma2.shape + (l_,))
    z = np.zeros(l_)

    # Compute correlation
    c2, sigma_vec = cov_2_corr(sigma2)

    for l in range(l_):
        lam = lambda_vec[l]

        # perform glasso shrinkage
        _, invs2_tilde, *_ = graphical_lasso(c2, lam)

        # correlation extraction
        c2_tilde = np.linalg.solve(invs2_tilde, np.eye(invs2_tilde.shape[0]))
        c2_bar[:, :, l] = cov_2_corr(c2_tilde)[0]  # estimated corr.

        # inv. corr.
        phi2_bar[:, :, l] = np.linalg.solve(c2_bar[:, :, l],
                                            np.eye(c2_bar[:, :, l].shape[0]))

        tmp = abs(phi2_bar[:, :, l])
        z[l] = np.sum(tmp < tol)

    # selection
    index = list(np.where(z >= k)[0])
    if len(index) == 0:
        index.append(l)
        conv = 0  # target of k null entries not reached
    else:
        conv = 1  # target of k null entries reached
    l_bar = index[0]
    lambda_bar = lambda_vec[l_bar]

    # output
    if not opt:
        c2_bar = c2_bar[:, :, l_bar]  # shrunk correlation
        phi2_bar = phi2_bar[:, :, l_bar]  # shrunk inverse correlation
        l_bar = None
        # shrunk covariance
        sigma2_bar = np.diag(sigma_vec) @ c2_bar @ np.diag(sigma_vec)
    else:
        sigma2_bar = np.zeros(sigma2.shape + (l_,))
        for l in range(l_):
            sigma2_bar[:, :, l] = np.diag(sigma_vec) @ c2_bar[:, :, l] @ \
                                  np.diag(sigma_vec)

    return sigma2_bar, c2_bar, phi2_bar, lambda_bar, conv, l_bar
