import numpy as np
from scipy.linalg import expm, logm
from cvxopt import matrix
from cvxopt.solvers import qp, options


def project_trans_matrix(p, delta_t):
    """For details, see here.

    Parameters
    ----------
        p : array, shape (c_, c_)
        delta_t : scalar

    Returns
    -------
        p_delta_t : array, shape (c_, c_)

    """

    c_ = len(p)

    # Step 1: Compute log-matrix

    l = logm(p)

    # Step 2: Compute generator

    P = matrix(np.eye(c_ * c_))
    q = matrix(-l.reshape((c_ * c_, 1)))
    G = matrix(0.0, (c_ * c_, c_ * c_))
    G[::c_ * c_ + 1] = np.append([0], np.tile(np.append(-np.ones(c_), [0]), c_ - 1))
    h = matrix(0.0, (c_ * c_, 1))
    A = matrix(np.repeat(np.diagflat(np.ones(c_)), c_, axis=1))
    b = matrix(0.0, (c_, 1))
    options['show_progress'] = False
    g = qp(P, q, G, h, A, b)['x']

    # Step 3: Compute projected transition matrix

    g = np.array(g).reshape((c_, c_))
    p_delta_t = expm(delta_t * g)

    p_delta_t[-1, :] = np.zeros((1, p.shape[1]))
    p_delta_t[-1, -1] = 1

    return p_delta_t
