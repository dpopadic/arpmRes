#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def adjusted_value(v_close, dates, cf_r, r, fwd=True):
    """For details, see here.

    Parameters
    ----------
    v_close : array, shape(t_,)
    dates : array, shape (t_,)
    cf_r : shape (k_,) optional
    r : shape (k_,) optional
    fwd : int, optional

    Returns
    -------
    v_adj : array, shape (t_,)

    """

    v_adj = np.copy(v_close)
    ind_r = []
    [ind_r.append(np.where(dates == x)[0][0]) for x in r]
    if fwd:
        # forward cash-flow-adjusted values
        if len(ind_r) > 0:
            for r_k, cf_rk in zip(ind_r, cf_r):
                v_adj[r_k:] = v_adj[r_k:] *\
                    (1 + cf_rk / (v_close[r_k] - cf_rk))
    else:
        # backward cash-flow-adjusted values
        if len(ind_r) > 0:
            for r_k, cf_rk in zip(ind_r, cf_r):
                v_adj[:r_k] = v_adj[:r_k] *\
                    (1 - cf_rk / (v_close[r_k]))

    return np.squeeze(v_adj)
