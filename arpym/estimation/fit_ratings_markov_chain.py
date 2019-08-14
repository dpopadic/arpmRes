#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from scipy.linalg import expm


def fit_ratings_markov_chain(dates, n_oblig, n_cum, tau_hl=None):
    """For details, see here.

    Parameters
    ----------
        dates : array, shape(t_,)
        n_oblig : array, shape (t_, c_)
        n_cum : array, shape (t_, c_, c_)
        tau_hl : scalar, optional

    Returns
    -------
        p : array, shape (c_, c_)

    """
    t_ = len(dates)
    c_ = n_oblig[-1].shape[0]
    delta_t = np.zeros(t_ - 1)

    # estimation
    num = np.zeros((c_, c_))
    den = np.zeros((c_, c_))
    g = np.zeros((c_, c_))

    # Step 1: Compute number of transitions at each time t

    m_num = np.zeros(n_cum.shape)
    m_num[0, :, :] = n_cum[0, :, :]
    m_num[1:, :, :] = np.diff(n_cum, axis=0)

    # Step 2: Compute generator

    for i in range(c_):
        for j in range(c_):
            if i != j:
                if tau_hl is None:  # ML
                    # numerator
                    num[i, j] = n_cum[-1, i, j]
                    # denominator
                    for t in range(1, t_):
                        den[i, j] = den[i, j] + n_oblig[t, i] * (np.busday_count(dates[t - 1], dates[t])) / 252
                    g[i, j] = num[i, j] / den[i, j]
                else:  # MLFP
                    # numerator and denominator
                    for t in range(t_):
                        num[i, j] = num[i, j] + m_num[t, i, j] * np.exp(
                            -(np.log(2) / tau_hl) * (np.busday_count(dates[t], dates[-1])) / 252)
                    for t in range(1, t_):
                        den[i, j] = den[i, j] + n_oblig[t - 1, i] * (
                                    np.exp(-(np.log(2) / tau_hl) * (np.busday_count(dates[t], dates[-1])) / 252)
                                    - np.exp(-(np.log(2) / tau_hl) * (np.busday_count(dates[t - 1], dates[-1])) / 252))

                    g[i, j] = (np.log(2) / tau_hl) * num[i, j] / den[i, j]

    for i in range(c_):
        g[i, i] = -np.sum(g[i, :])

    # Step 3: Compute transition matrix

    p = expm(g)

    return p
