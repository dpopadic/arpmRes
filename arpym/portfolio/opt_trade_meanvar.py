# -*- coding: utf-8 -*-

import numpy as np


def opt_trade_meanvar(h_start, h_end, q_bar, alpha, beta, eta, gamma,
                      sigma, delta_q):
    """For details, see here.

    Parameters
    ----------
        h_start : array, shape (j_,)
        h_end : array, shape (j_,)
        q_bar : scalar
        alpha : scalar
        beta : scalar
        eta : scalar
        gamma : scalar
        sigma : scalar
        delta_q : scalar

    Returns
    -------
        mu_pi : array, shape (j_,)
        sig2_pi : array, shape (j_,)

    """

    xi = beta ** (alpha + 1) / (beta + beta * alpha - alpha)
    e_pi = q_bar * (gamma / 2 * (h_end ** 2 - h_start ** 2) -
                    eta * xi * np.abs(h_end - h_start) ** (1 + alpha) *
                    delta_q ** (- alpha))
    v_pi = (q_bar * sigma) ** 2 * delta_q * \
           (h_start ** 2 + 2 * h_start * (h_end - h_start) / (beta + 1) +
            (h_end - h_start) ** 2 / (2 * beta + 1))

    return e_pi, v_pi
