# -*- coding: utf-8 -*-
import numpy as np

from arpym.statistics.simulate_normal import simulate_normal
from arpym.statistics.twist_prob_mom_match import twist_prob_mom_match
from arpym.tools.transpose_square_root import transpose_square_root


def simulate_quadn(alpha, beta, gamma, mu, sigma2, j_):
    """For details, see here.

    Parameters
    ----------
        alpha : float
        beta : array, shape(n_,)
        gamma : array, shape(n_, n_)
        mu : array, shape(n_,)
        sigma2 : array, shape(n_, n_)
        j_ : float

    Returns
    -------
        y : array, shape(j_,)
        p_ : array, shape(j_,)
    """
    if np.ndim(beta) == 1:
        n_ = len(beta)
    else:
        n_ = 1

    beta, gamma = np.reshape(beta, (n_, 1)), np.reshape(gamma, (n_, n_))
    mu, sigma2 = np.reshape(mu, (n_, 1)), np.reshape(sigma2, (n_, n_))

    # Step 1: Perform cholesky decomposition of sigma2

    l = transpose_square_root(sigma2, 'Cholesky')

    # Step 2: Compute parameter lambda

    lambd, e = np.linalg.eig(l.T @ gamma @ l)
    lambd = lambd.reshape(-1, 1)

    # Step 3: Compute new parameters beta_tilde and gamma_tilde

    beta_tilde = beta + 2 * gamma @ mu
    gamma_tilde = e.T @ l.T @ beta_tilde

    # Step 4: Generate Monte Carlo scenarios

    z = simulate_normal(np.zeros(n_), np.eye(n_), j_).reshape(-1, n_)
    y = alpha + beta.T @ mu + mu.T @ gamma @ mu + \
        z @ gamma_tilde + z ** 2 @ lambd

    # Step 5: Match expectation and variance by twisting probabilities

    e_y = alpha + beta.T @ mu + mu.T @ gamma @ mu + np.trace(sigma2 @ gamma)
    v_y = 2 * np.trace(np.linalg.matrix_power(gamma @ sigma2, 2)) + \
          beta_tilde.T @ sigma2 @ beta_tilde
    p_ = twist_prob_mom_match(y.reshape(-1, 1), e_y[0], v_y).reshape(-1)

    return np.squeeze(y), np.squeeze(p_)
