# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from scipy import interpolate

from arpym.pricing.bsm_function import bsm_function
from arpym.pricing.shadowrates_ytm import shadowrates_ytm


def call_option_value(x_s, x_y, tau_y, x_sigma, m_moneyness, tau_implvol,
                      k_strk, t_end, t_hor):
    """For details, see here.

    Parameters
    ----------
        x_s : scalar
        x_y : scalar
        x_sigma : array, shape(k_*l_,) or shape(k_, l_)
        tau_implvol : shape(k_,)
        m_moneyness : shape(l_,)
        k_strk : scalar
        t_end : date
        t_hor : date

    Return
    ------
        v : scalar


    """

    x_sigma = x_sigma.reshape(-1)

    # Step 1: Compute time to expiry of the call option at thor

    tau = np.busday_count(t_hor, t_end)/252

    # Step 2: Compute value of the underlying

    s = np.exp(x_s)

    # Step 3: Compute m moneyness

    m = np.log(s/k_strk)/np.sqrt(tau)

    # Step 4: Compute shadow yield

    if x_y.shape[0] == 1:
        x_y_interp = x_y
    else:
        interp = sp.interpolate.interp1d(tau_y.flatten(), x_y, axis=1,
                                         fill_value='extrapolate')
        x_y_interp = interp(tau)

    # Step 5: Compute discount rate

    y = shadowrates_ytm(x_y_interp)

    # Step 6: Compute log-implied volatility in point (m, tau)

    points = list(zip(*[grid.flatten() for grid in
                        np.meshgrid(*[tau_implvol, m_moneyness])]))
    m_e = min(max(m, min(m_moneyness)), max(m_moneyness))  # extrapolation
    tau_e = min(max(tau, tau_implvol[0]), tau_implvol[-1])  # extrapolation
    x_sigma_interp = \
        interpolate.LinearNDInterpolator(points, x_sigma)(*np.r_[tau_e, m_e])

    # Step 7: Compute implied volatility

    sigma_interp = np.exp(x_sigma_interp)

    # Step 8: Compute call option value

    v = bsm_function(s, y, sigma_interp, m, tau)

    return v
