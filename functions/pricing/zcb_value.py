# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp

from arpym.pricing.nelson_siegel_yield import nelson_siegel_yield
from arpym.pricing.shadowrates_ytm import shadowrates_ytm


def zcb_value(t_hor, t_end, rd_type, x_thor, *, tau_x=None,
              facev=1, eta=0.013):
    """For details, see here.

    Parameters
    ----------
        t_hor : date
        t_end : array, shape(k_,)
        rd_type : string,
        x_thor : array, shape(j_, d_) if d_>1 or (t_,) for d_=1
        tau_x : array, optional, shape(d_,)
        facev : array, optional, shape(k_,)
        eta : scalar

    Returns
    -------
        v : array, shape(j_, k_) if k_>1 or (t_,) for k_=1

    """

    j_ = x_thor.shape[0]
    k_ = t_end.shape[0]

    if isinstance(facev, int):
        facev = facev*np.ones(k_)

    # Step 1: Compute times to maturity at the horizon

    tau_hor = np.array([np.busday_count(t_hor, t_end[i])/252
                        for i in range(k_)])

    # Step 2: Compute the yield to maturity

    if rd_type == 'y':

        # Step 2a: Interpolate yields

        interp = sp.interpolate.interp1d(tau_x.flatten(), x_thor, axis=1,
                                         fill_value='extrapolate')
        y = interp(tau_hor)

    elif rd_type == 'sr':

        # Step 2b: Compute yields

        # interpolate shadow rates
        interp = sp.interpolate.interp1d(tau_x.flatten(), x_thor, axis=1,
                                         fill_value='extrapolate')
        # Transform shadow rates to yields
        y = shadowrates_ytm(interp(tau_hor), eta)

    elif rd_type == 'ns':

        # Step 2c: Compute yields

        y = np.zeros((j_, k_))
        idx_nonzero = (tau_hor > 0)
        for j in range(j_):
            y[j, idx_nonzero] = nelson_siegel_yield(tau_hor[idx_nonzero],
                                                    x_thor[j])

    # Step 3: Compute zero coupon-bonds value

    v = facev*np.exp(-tau_hor * y)

    return np.squeeze(v)
