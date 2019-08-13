#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict
from scipy.optimize import newton

from arpym.pricing import fit_nelson_siegel_bonds
from arpym.pricing import nelson_siegel_yield


def bootstrap_nelson_siegel(v_bond, dates, c, tend, freq_paym=0.5, facev=1.0):
    """For details, see here.

    Parameters
    ----------
        v_bond : array, shape (t_, n_)
        dates : array, shape (t_, )
        c : array, shape (n_,)
        tend : array, shape (n,)
        freq_paym : scalar
        tau : array, optional, shape (l_)
        tau_ref : array, optional, shape (m_)
        y_ref : array, optional, shape (t_, m_)

    Returns
    ----------
        theta : array, shape (t_, 4)
        y_tau : array, shape (t_, l_)
        y_ref_tau : array, shape (t_, l_)
        s_tau : array, shape (t_, l_)

    """

    n_ = v_bond.shape[1]
    t_ = len(dates)

    theta = np.zeros((t_, 4))

    for t in range(t_):
        tau_real = np.zeros(n_)
        c_k = defaultdict(dict)
        upsilon = defaultdict(dict)
        for n in range(n_):
            # Step 1: Compute time to maturity

            # time from dates[t] to the maturity (in years)
            tau_real[n] = np.busday_count(dates[t], tend[n]) / 252

            # Step 2: Compute the number of coupon payments, time to coupon payments, and coupons

            # time to coupon payments
            upsilon[n] = np.flip(np.arange(tau_real[n], 0.0001, -freq_paym), 0)
            # number of coupon payments from dates[t] to maturity
            k_n = len(upsilon[n])
            # every bond has 1/freq_paym payments per year
            c_k[n] = c[n] * np.ones(k_n) * freq_paym
            # include notional
            c_k[n][-1] = c_k[n][-1] + 1

        # Step 3: Fit NS model

        if t == 0:
            theta[t] = fit_nelson_siegel_bonds(v_bond[t, :], c_k, upsilon, facev=facev)
        else:
            theta[t] = fit_nelson_siegel_bonds(v_bond[t, :], c_k, upsilon,
                                               theta_0=theta[t - 1], facev=facev)

    return theta
