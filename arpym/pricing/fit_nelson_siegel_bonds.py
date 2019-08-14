#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize

from arpym.pricing import nelson_siegel_yield


def fit_nelson_siegel_bonds(v_bond, c, upsilon, *, facev=1, theta_0=None):
    """For details, see here.

    Parameters
    ----------
        v_bond : array, shape (n_,)
        c : dict, length (n_)
        upsilon : dict, length (n_)
        facev : scalar
        theta_0: array, shape (4,)

    Returns
    -------
        theta : array, shape (4,)

    """
    n_ = len(v_bond)

    def fit_nelsonsiegel_bonds_target(theta):
        v_bond_theta = np.zeros(n_)
        output = 0.0
        for n in range(n_):

            # Step 1: Compute Nelson-Siegel yield curve

            y_theta = nelson_siegel_yield(upsilon[n], theta)

            # Step 2: Compute coupon bond value

            # zero-coupon bond value
            v_zcb = np.exp(-upsilon[n] * y_theta)
            # bond value
            v_bond_theta[n] = facev * (c[n] @ v_zcb)

            # Step 3: Compute minimization function

            if n == 0:
                h_tilde = (upsilon[n + 1][-1] - upsilon[n][-1]) / 2
            elif n == n_ - 1:
                h_tilde = (upsilon[n][-1] - upsilon[n - 1][-1]) / 2
            else:
                h_tilde = (upsilon[n + 1][-1] - upsilon[n - 1][-1]) / 2
            output += h_tilde * np.abs(v_bond_theta[n] - v_bond[n])
        return output

    if theta_0 is None:
        theta_0 = 0.1 * np.ones(4)

    # Step 4: Fit Nelson-Siegel parameters

    res = minimize(fit_nelsonsiegel_bonds_target, theta_0,
                   bounds=((None, None), (None, None), (None, None), (0, None)))
    theta = res.x
    # Output
    return theta
