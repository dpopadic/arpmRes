# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize
from scipy.misc import logsumexp
from scipy.sparse import eye

from arpym.statistics import meancov_sp

def min_rel_entropy_sp(p_pri, z_ineq=None, mu_view_ineq=None, z_eq=None, mu_view_eq=None,
                       normalize=True):
    """For details, see here.

    Note
    ----
        The constaints :math:`p_j \geq 0` and :math:`\sum p_j = 1` are set
        automatically.

    Parameters
    ----------
        p_pri : array, shape(j_,)
        z_ineq : array, shape(l_, j_), optional
        mu_view_ineq : array, shape(l_,), optional
        z_eq : array, shape(m_, j_), optional
        mu_view_eq : array, shape(m_,), optional
        normalize : bool, optional

    Returns
    -------
        p_ : array, shape(j_,)
    """

    # Step 1: Concatenate the constraints and concatenated constraints

    if z_ineq is None and z_eq is None:
        # if there is no constraint, then just return p_pri
        return p_pri
    elif z_ineq is None:
        # no inequality constraints
        z = z_eq
        mu_view = mu_view_eq
        l_ = 0
        m_ = len(mu_view_eq)
    elif z_eq is None:
        # no equality constraints
        z = z_ineq
        mu_view = mu_view_ineq
        l_ = len(mu_view_ineq)
        m_ = 0
    else:
        z = np.concatenate((z_ineq, z_eq), axis=0)
        mu_view = np.concatenate((mu_view_ineq, mu_view_eq), axis=0)
        l_ = len(mu_view_ineq)
        m_ = len(mu_view_eq)

    if normalize is True:
        # normalize the constraints
        m_z, s2_z = meancov_sp(z.T)
        s_z = np.sqrt(np.diag(s2_z))
        z = ((z.T - m_z) / s_z).T
        mu_view = (mu_view - m_z) / s_z

    # Step 2: Compute the Lagrange dual function, gradient and Hessian

    # pdf of a discrete exponential family
    def exp_family(theta):
        x = theta @ z + np.log(p_pri)
        phi = logsumexp(x)
        p = np.exp(x - phi)
        p[p < 1e-32] = 1e-32
        p = p / np.sum(p)
        return p

    # minus dual Lagrangian
    def lagrangian(theta):
        x = theta @ z + np.log(p_pri)
        phi = logsumexp(x)  # stable computation of log sum exp
        return phi - theta @ mu_view

    def gradient(theta):
        return z @ exp_family(theta) - mu_view

    def hessian(theta):
        p = exp_family(theta)
        z_ = z.T - z @ p
        return (z_.T * p) @ z_

    # Step 3: Compute optimal Lagrange multipliers and the posterior probabilities

    k_ = l_ + m_  # dimension of the Lagrange dual problem
    theta0 = np.zeros(k_)  # intial value

    if l_ == 0:
        # if no constraints, then perform the Newton conjugate gradient
        # trust-region algorithm
        options = {'gtol': 1e-10}
        res = minimize(lagrangian, theta0, method='trust-ncg',
                       jac=gradient, hess=hessian, options=options)
    else:
        # otherwise perform sequential least squares programming
        options = {'ftol': 1e-10, 'disp': False, 'maxiter': 1000}
        alpha = -eye(l_, k_)
        constraints = {'type': 'ineq',
                       'fun': lambda theta: alpha @ theta}
        res = minimize(lagrangian, theta0, method='SLSQP', jac=gradient,
                       constraints=constraints, options=options)

    return np.squeeze(exp_family(res['x']))
