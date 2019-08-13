# -*- coding: utf-8 -*-
import numpy as np

from arpym.statistics.ewm_meancov import ewm_meancov


def smoothing(s, tau_hl, w=None):
    """For details, see here.

    Parameters
    ----------
        s : array, shape (t_,)
        tau_hl : scalar
        w : int

    Returns
    -------
        s_smooth : array, shape (t_-w+1,)

    """

    t_ = s.shape[0]

    if w is None:
        s_smooth = np.zeros(t_)
        for t in range(t_):
            s_smooth[t] = ewm_meancov(s, tau_hl, t+1, t+1)[0]

    else:
        s_smooth = np.zeros(t_-w+1)
        for t in range(w-1, t_):
            s_smooth[t-w+1] = ewm_meancov(s, tau_hl, t+1, w)[0]

    return np.squeeze(s_smooth)
