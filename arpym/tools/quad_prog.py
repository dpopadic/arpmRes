# -*- coding: utf-8 -*-
import numpy as np
from cvxopt import matrix, sparse, spmatrix
from cvxopt.solvers import qp


def quad_prog(H, f, aeq, beq, lb=None, ub=None, x0=None):
    """For details, see here.

    Minimize:
            (1/2)*x'*H*x + f'*x
    subject to:
            aeq*x = beq
            lb <= x <= ub

    """

    if lb is None:
        lb = -100 * np.ones((f.shape[0], 1))
    if ub is None:
        ub = 100 * np.ones((f.shape[0], 1))
    P, q, G, h, A, b = _convert(H, f, aeq, beq, lb, ub)
    results = qp(P, q, G, h, A, b)

    # Convert back to NumPy matrix
    # and return solution
    xstar = results['x']
    return np.asarray(xstar)


def _convert(H, f, aeq, beq, lb, ub):
    """For details, see here.

    Convert everything to cvxopt-style matrices

    """

    P = matrix(H)
    q = matrix(f.astype(np.double))
    if aeq is None:
        A = None
    else:
        A = matrix(aeq.astype(np.double))
    if beq is None:
        b = None
    else:
        b = matrix(beq.astype(np.double))

    n = lb.size
    G = sparse([-speye(n), speye(n)])
    h = matrix(np.vstack([-lb.astype(np.double), ub.astype(np.double)]))
    return P, q, G, h, A, b


def speye(n):
    """For details, see here.

    """
    r = range(n)
    return spmatrix(1.0, r, r)
