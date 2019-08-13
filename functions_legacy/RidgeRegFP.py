from numpy import ones, zeros, eye, tile, r_
from numpy.linalg import pinv

from FPmeancov import FPmeancov


def RidgeRegFP(X, Z, p, lam):
    # This function computes the solutions of the ridge regression of X on Z
    #  INPUTS
    # X        :[matrix](n_ x t_end) time series of target observations
    # Z        :[matrix](k_ x t_end) time series of factors observations
    # p        :[vector](1 x t_end) Flexible Probabilities
    # lam   :[vector](1 x l_) penalties for ridge regression
    #  OPS
    # alpha_l  :[matrix](n_ x l_) shifting parameter
    # beta_l   :[array](n_ x k_ x l_) array of optimal loadings
    # s2_l     :[array](n_ x n_ x l_) covariance matrix of residuals
    # U        :[array](n_ x t_end x l_) time series of residuals

    ## Code

    n_, t_ = X.shape
    k_ = Z.shape[0]
    l_ = len(lam)

    # if p are not provided, observations are equally weighted
    if p is None:
        p = (1 / t_) @ ones((1, t_))

    # compute HFP mean and covariance of joint variable (XZ)
    m_joint, s2_joint = FPmeancov(r_[X, Z], p)
    m_X = m_joint[:n_]
    m_Z = m_joint[n_:n_ + k_ + 1]
    s2_XZ = s2_joint[:n_, n_:n_ + k_ + 1]
    s2_Z = s2_joint[n_:n_ + k_ + 1, n_:n_ + k_ + 1]

    alpha_l = zeros((n_, l_))
    beta_l = zeros((n_, k_, l_))
    s2_l = zeros((n_, n_, l_))
    U = zeros((n_, t_, l_))
    # compute solutions for every penalty
    for l in range(l_):
        beta_l[:, :, l] = s2_XZ.dot(pinv(s2_Z + lam[l] * eye(k_)))
        alpha_l[:, l] = m_X - beta_l[:, :, l] @ m_Z
        U[:, :, l] = X - tile(alpha_l[:, l], (1, t_)) - beta_l[:, :, l] @ Z
        [_, s2_l[:, :, l]] = FPmeancov(U[:, :, l], p)

    return alpha_l, beta_l, s2_l, U
