# -*- coding: utf-8 -*-
import numpy as np

from scipy.integrate import quad

def spectral_index(spectr, pi, p, h_tilde):
    """For details, see here.

    Parameters
    ----------
        spectr : function
        pi : array, shape (j_, n_)
        p : array,  shape (j_, )
        h_tilde : array, shape (n_, )

    Returns
    -------
        satis_spectr : scalar
        satis_spectr_grad : scalar

    """

    h_tilde = np.array(h_tilde).reshape(-1)
    p = np.array(p).reshape(-1)

    j_ = pi.shape[0]

    # Step 1: compute the sorted ex-ante performance scenarios

    # compute the ex-ante performance scenarios
    y = pi@h_tilde

    # sort ex-ante performance scenarios, probabilities and P&Ls
    sort_j = np.argsort(y)
    y_sort = y[sort_j]
    p_sort = p[sort_j]
    pi_sort = pi[sort_j, :]

    # compute the cumulative sums of the ordered probabilities
    u_sort = np.append(0, np.cumsum(p_sort))

    # Step 2: compute the weights of spectral measure

    w = np.zeros(j_)
    for j in range(j_):
        w[j], _ = quad(spectr, u_sort[j], u_sort[j + 1])
    w = w / np.sum(w)

    # Step 3: compute the spectral measure

    satis_spectr = y_sort@w

    # Step 4: compute the gradient of the spectral measure

    satis_spectr_grad = pi_sort.T@w

    return satis_spectr, satis_spectr_grad
