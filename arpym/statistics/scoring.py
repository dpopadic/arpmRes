# -*- coding: utf-8 -*-
import numpy as np

from arpym.statistics.ewm_meancov import ewm_meancov


def scoring(s, tau_hl, w=None):
    """For details, see here.

    Parameters
    ----------
        s : array, shape (t_,)
        tau_hl : scalar
        w : int

    Returns
    -------
        s_score : array, shape (t_-w+1,)

    """

    t_ = s.shape[0]

    if w is None:
        s_score = np.zeros(t_)
        for t in range(1, t_):
            ewma_t, ewm_cv_t = ewm_meancov(s, tau_hl, t+1, t+1)
            ewm_sd_t = np.sqrt(ewm_cv_t)
            s_score[t] = (s[t]-ewma_t)/ewm_sd_t

    else:
        s_score = np.zeros(t_-w+1)
        for t in range(w-1, t_):
            ewma_t, ewm_cv_t = ewm_meancov(s, tau_hl, t+1, w)
            ewm_sd_t = np.sqrt(ewm_cv_t)
            s_score[t-w+1] = (s[t]-ewma_t)/ewm_sd_t

    return np.squeeze(s_score)
