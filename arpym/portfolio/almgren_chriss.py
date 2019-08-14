w#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


def almgren_chriss(q, h_start, h_end, lam, eta, sigma, theta=None):
    """For details, see here.

    Parameters
    ----------
        q : array, shape (k_,)
        h_start : scalar
        h_end : scalar
        lam : scalar
        eta : scalar
        sigma : scalar
        theta : scalar

    Returns
    -------
        h_q : array, shape (k_,)
        hspeed_q : array, shape (k_,)

    """

    if lam != 0:
        coeff2 = 2*lam*sigma**2
        if theta is None:
            theta = coeff2 * h_end
        coeff1 = np.sqrt(lam/eta)*sigma
        coeff = np.sinh(coeff1*(q[-1]-q[0]))
        h_q = (h_start-theta/coeff2)*np.sinh(coeff1*(q[-1]-q))/coeff +\
              (h_end-theta/coeff2)*np.sinh(coeff1*(q-q[0]))/coeff+theta/coeff2
        hspeed_q = -coeff1*(h_start-theta/coeff2)*np.cosh(coeff1*(q[-1]-q)) /\
            coeff+coeff1*(h_end-theta/coeff2)*np.cosh(coeff1*(q-q[0]))/coeff
    else:
        h_q = q*(h_end-h_start)/(q[-1]-q[0]) +\
            (h_start*q[-1]-h_end*q[0])/(q[-1]-q[0])
        hspeed_q = (h_end-h_start)/(q[-1]-q[0])*np.ones((1, len(q)))

    return h_q, hspeed_q
