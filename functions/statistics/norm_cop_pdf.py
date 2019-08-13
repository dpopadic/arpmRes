# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import solve, norm, det
import scipy.stats as stats


def norm_cop_pdf(u, mu, sigma2):
    """For details, see here.

    Parameters
    ----------
        u :  array, shape (n_,)
        mu : array, shape (n_,)
        sigma2 :  array, shape (n_, n_)

    Returns
    -------
        pdf_u : scalar

    """

    # Step 1: Compute the inverse marginal cdf's

    svec = np.sqrt(np.diag(sigma2))
    x = stats.norm.ppf(u.flatten(), mu.flatten(), svec).reshape(-1, 1)

    # Step 2: Compute the joint pdf

    n_ = len(u)
    pdf_x = (2*np.pi)**(-n_ / 2)*((det(sigma2))**(-.5))*np.exp(-0.5 * (x - mu.reshape(-1, 1)).T@(solve(sigma2, x - mu.reshape(-1, 1))))

    # Step 3: Compute the marginal pdf's

    pdf_xn = stats.norm.pdf(x.flatten(), mu.flatten(), svec)

    # Compute the pdf of the copula
    pdf_u = np.squeeze(pdf_x / np.prod(pdf_xn))

    return pdf_u
