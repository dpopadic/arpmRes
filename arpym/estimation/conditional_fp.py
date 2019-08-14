# -*- coding: utf-8 -*-

import numpy as np

from arpym.estimation.crisp_fp import crisp_fp
from arpym.views.min_rel_entropy_sp import min_rel_entropy_sp


def conditional_fp(z, z_star, alpha, p_prior):
    """For details, see here.

    Parameters
    ----------
        z : array, shape (t_, )
        z_star : array, shape (k_, )
        alpha : scalar
        p_prior : array, shape (t_, )

    Returns
    -------
        p : array, shape (t_, k_) if k_>1 or (t_,) for k_=1

    """

    z_star = np.atleast_1d(z_star)

    t_ = z.shape[0]
    k_ = z_star.shape[0]

    # Step 1: Compute crisp probabilities

    p_crisp, _, _ = crisp_fp(z, z_star, alpha)
    p_crisp = p_crisp.reshape(k_, t_)
    p_crisp[p_crisp == 0] = 10**-20

    for k in range(k_):
        p_crisp[k, :] = p_crisp[k, :]/np.sum(p_crisp[k, :])

    # Step 2: Compute conditional flexible probabilities

    p = np.zeros((k_, t_))

    for k in range(k_):
        # moments
        m_z = p_crisp[k, :]@z
        s2_z = p_crisp[k, :]@(z**2)-m_z**2

        # constraints
        a_ineq = np.atleast_2d(z**2)
        b_ineq = np.atleast_1d((m_z**2)+s2_z)
        a_eq = np.array([z])
        b_eq = np.array([m_z])

        # output
        p[k, :] = min_rel_entropy_sp(p_prior, a_ineq, b_ineq, a_eq, b_eq)

    return np.squeeze(p.T)
