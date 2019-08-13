# -*- coding: utf-8 -*-
import numpy as np

from cvxopt import matrix
from cvxopt import solvers


def obj_tracking_err(s2_x_xb, s=None):
    """For details, see here.

    Parameters
    ----------
        s2_x_xb : array, shape (n_ + 1, n_ + 1)
        s : array, shape (k_, ) or int

    Returns
    -------
    w_star : array, shape (n_, )
    minus_te : scalar

    """
    # read the number of components in x
    n_ = np.shape(s2_x_xb)[0] - 1

    # shift indices of instruments by -1
    if s is None:
        s = np.arange(n_)
    elif np.isscalar(s):
        s = np.array([s - 1])
    else:
        s = s - 1

    ## Step 0: LCQP optimization setup

    # quadratic objective parameters
    s2_0 = s2_x_xb[:-1, :-1]
    u_0 = -(s2_x_xb[:-1, -1].reshape(-1, 1))
    v_0 = s2_x_xb[-1, -1]

    # linear constraint parameters
    c_sort = np.array([k for k in np.arange(0, n_) if k not in list(s)])

    if c_sort.size == 0:
        a_1 = np.ones((1, n_))
    else:
        # first row of a_1
        first_r = np.ones((1, n_))
        first_r[0, c_sort] = 0
        # rest rows of a_1
        rest_r = (np.eye(n_))[c_sort, :]
        a_1 = np.concatenate((first_r, rest_r))

    a_2 = np.zeros((c_sort.size + 1, 1))
    a_2[0, 0] = 1

    b_1 = np.eye(n_)
    b_2 = np.zeros((n_, 1))

    ## step 1: perform optimization using CVXOPT function solver.coneqp for LCQP

    # prepare data types for CVXPOT
    P = matrix(s2_0, tc='d')
    q = matrix(u_0, tc='d')
    A = matrix(a_1, tc='d')
    b = matrix(a_2, tc='d')
    G = matrix(-b_1, tc='d')
    h = matrix(b_2, tc='d')

    # run optimization function
    solvers.options['show_progress'] = False
    sol = solvers.coneqp(P, q, G, h, A=A, b=b)

    # prepare output
    w_star = np.array(sol['x'])
    minus_te = -(np.sqrt(w_star.T @ s2_0 @ w_star + 2 * w_star.T @ u_0 + v_0))

    return w_star, np.asscalar(minus_te.reshape(-1))
