# -*- coding: utf-8 -*-

import numpy as np
from arpym.estimation.fit_locdisp_mlfp import fit_locdisp_mlfp
from arpym.estimation.fit_lfm_mlfp import fit_lfm_mlfp
from arpym.tools.mahalanobis_dist import mahalanobis_dist


def fit_locdisp_mlfp_difflength(epsi, p=None, nu=4., *, threshold=10**(-5),
                     maxiter=10**5):
    """For details, see here.

    Parameters
    ----------
        epsi : array, shape (t_,)
        p : array, shape (t_,), optional
        nu : float, optional
        threshold: float, optional
        maxiter : float

    Returns
    -------
        mu : float
        sigma2 : float

    Note: We suppose the missing values, if any, are at the beginning (the
    farthest observations in the past could be missing).

    """

    if isinstance(threshold, float):
        threshold = [threshold, threshold]

    t_, i_ = epsi.shape

    if p is None:
        p = np.ones(t_)/t_

    # Step 0: Initialize

    # Reshuffle the series in a nested pattern, such that the series with the
    # longer history comes first and the one with the shorter history comes
    # last.

    l_ = np.zeros(i_)
    for i in range(i_):
        l_[i] = min(np.where(~np.isnan(epsi[:, i]))[0])

    index = np.argsort(l_)
    l_sort = l_[index]

    epsi_sort = epsi[:, index]
    idx = np.argsort(index)

    c = 0
    epsi_nested = []
    epsi_nested.append(epsi_sort[:, 0])
    t = []
    t.append(int(l_sort[0]))
    for j in range(1, i_):
        if l_sort[j] == l_sort[j-1]:
            epsi_nested[c] = np.column_stack((epsi_nested[c],
                                              epsi_sort[:, j]))
        else:
            c = c+1
            epsi_nested.append(epsi_sort[:, j])
            t.append(int(l_sort[j]))

    c_ = len(epsi_nested)

    mu, sig2 = fit_locdisp_mlfp(epsi_nested[0], p=p, nu=nu, threshold=threshold[0], maxiter=maxiter)
    if np.ndim(epsi_nested[0]) > 1:
        ii_ = epsi_nested[0].shape[1]
    else:
        ii_ = 1
    mu, sig2 = mu.reshape((ii_, 1)), sig2.reshape((ii_, ii_))

    # Step 1: maximize conditional log-likelihoods

    for c in range(1, c_):
        data = epsi_nested[c][t[c]:]
        e = np.zeros((t_-t[c], mu.shape[0]))
        sza = 1
        for j in range(c):
            if epsi_nested[j].ndim == 2:
                szb = epsi_nested[j].shape[1]
                e[:, sza-1:sza-1+szb] = epsi_nested[j][t[c]:t_, :]
            else:
                szb = 1
                e[:,
                  sza-1:sza-1+szb] = np.atleast_2d(epsi_nested[j][t[c]:t_]).T

            sza = sza+szb

        # a) probabilities
        p_k = p[t[c]:t_]/np.sum(p[t[c]:t_])

        # b) degrees of freedom
        nu_c = nu+e.shape[1]

        # c) loadings
        alpha, beta, s2, _ = fit_lfm_mlfp(data, e, p=p_k, nu=nu_c,
                                          tol=threshold[1], maxiter=maxiter)
        if np.ndim(data) < 2:
            n_ = 1
        else:
            n_ = data.shape[1]
        if np.ndim(e) < 2:
            k_ = 1
        else:
            k_ = e.shape[1]
        alpha, beta, s2 = alpha.reshape((n_, 1)), beta.reshape((n_, k_)), s2.reshape((n_, n_))

        # d) location/scatter
        mah = mahalanobis_dist(e[-1, :].reshape(1, -1), mu.reshape(-1), sig2).reshape(-1)
        gamma = (nu_c/(nu+mah))*s2+beta@sig2@beta.T
        sig2 = np.r_[np.r_['-1', sig2, sig2@beta.T],
                     np.r_['-1', beta@sig2, gamma]]
        sig2 = (sig2+sig2.T)/2
        mu = np.r_[mu, alpha+beta@mu]

    # Step 2: Output

    # reshuffling output
    mu = mu[idx]
    sig2 = sig2[np.ix_(idx, idx)]

    return np.squeeze(mu), np.squeeze(sig2)
