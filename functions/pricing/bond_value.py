# -*- coding: utf-8 -*-

import numpy as np
from arpym.pricing.zcb_value import zcb_value


def bond_value(t_hor, c, r, rd_type, x_thor, *, tau_x=None, facev=1,
               eta=0.013):
    """For details, see here .

    Parameters
    ----------
        t_hor : date
        c : array, shape(k_,)
        r : array, shape(k_,)
        rd_type : string, optional
        x_thor : array, shape(j_, d_)
        tau : array, shape(k_,)
        facev : scalar, optional
        eta : scalar, optional

    Returns
    -------
        v : array, shape(j_,)

    """

    # Step 0: Consider only coupons after the horizon time

    c = c[r >= t_hor]
    r = r[r >= t_hor]

    # Step 1: compute scenarios for coupon bond value

    # compute zero-coupon bond value
    v_zcb = zcb_value(t_hor, r, rd_type, x_thor, tau_x=tau_x, eta=eta)

    # include notional
    c[-1] = c[-1] + 1

    # compute coupon bond value
    if np.ndim(v_zcb) == 1:
        v_zcb = v_zcb.reshape(1, v_zcb.shape[0])
    v = facev * (v_zcb @ c)

    return v.reshape(-1)
