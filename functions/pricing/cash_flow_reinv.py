# -*- coding: utf-8 -*-

import numpy as np


def cash_flow_reinv(c, r, t_m, invfact):
    """For details, see here.

    Parameters
    ----------
        c : array, shape (k_,)
        r : array, shape (k_,)
        t_m : array, shape (m_+1,)
        invfact : array, shape (j_, m_) if m_>1 or (j_,) for m_=1

    Returns
    -------
        cf_tnow_thor : array, shape(j_, m_)

    """
    if len(invfact.shape) == 1:
        j_, m_ = invfact.shape[0], 1
    else:
        j_, m_ = invfact.shape
    k_ = r.shape[0]

    cf = np.zeros((j_, m_, k_))

    # Step 0: Find monit. time indexes corresponding to the coupon pay. dates

    ml = np.array([np.where(t_m == rc) for rc in r]).reshape(-1)
    ml = np.array([mll for mll in ml if mll.size != 0]).reshape(-1)
    l_ = len(ml)

    for l in np.arange(l_):

        # Step 1: Compute reinvestment factors from each payment day

        m_l = ml[l]
        if m_l != m_:
            invfact_tmk = invfact[:, m_l:]
        else:
            invfact_tmk = np.ones((j_, 1))

        # Step 2: Compute scenarios for the cumulative cash-flow path

        cf[:, m_l:, l] = c[l] * np.cumprod(invfact_tmk, axis=1)

    # compute cumulative reinvested cash-flow stream
    cf_tnow_thor = np.sum(cf, 2)

    return np.squeeze(cf_tnow_thor)
