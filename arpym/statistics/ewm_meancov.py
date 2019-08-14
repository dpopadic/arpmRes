# -*- coding: utf-8 -*-

import numpy as np

from arpym.statistics.meancov_sp import meancov_sp


def ewm_meancov(x, tau_hl, t=None, w=None):
    """For details, see here.

    Parameters
    ----------
        x : array, shape (t_, n_) if n_>1 or (t_,) for n=1
        tau_hl: scalar
        t : int
        w : int

    Returns
    -------
        ewma_t_x : array, shape (n_,)
        ewm_cv_t_x : array, shape (n_, n_)

    """
    t_ = x.shape[0]
    x = x.reshape(t_, -1)

    if t is None:
        t = t_
    if w is None:
        w = t_

    assert (t >= w), "t should be greater or equal to w."

    p_w = np.exp(-np.log(2)/tau_hl*np.arange(0, w))[::-1].reshape(-1)
    gamma_w = np.sum(p_w)
    ewma_t_x, ewm_cv_t_x = meancov_sp(x[t-w:t, :], p_w/gamma_w)

    return np.squeeze(ewma_t_x), np.squeeze(ewm_cv_t_x)
