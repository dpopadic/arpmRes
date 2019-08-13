from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from numpy import array, ones, zeros, abs, sqrt, tile
from numpy import max as npmax
from numpy.linalg import pinv

from FPmeancov import FPmeancov
from OrdLeastSquareFPNReg import OrdLeastSquareFPNReg
from SmartInverse import SmartInverse


def LassoRegFP(X, Z, p, lam, smartinverse=0):
    # Weighted lasso regression function
    # Note: LassoRegFP includes the function solveLasso created by GAUTAM V. PENDSE, http://www.gautampendse.com)
    # We made some changes in PENDSE's function in order to adapt it with SYMMYS's notation
    # The changes are made in conformity with the Creative Commons Attribution 3.0 Unported License
    #  INPUTS
    # x       :[matrix](n_ x t_end) time series of market observations
    # z       :[matrix](k_ x t_end) time series of factors
    # p       :[vector](t_end x 1) flexible probabilities
    # lambda  :[vector](l_ x 1) vector of penalties
    #  OP
    # alpha   :[matrix](n_ x l_) shifting term
    # beta    :[array](n_ x k_ x l_) loadings
    # s2_U    :[array](n_ x n_ x l_) covariance of residuals
    # U       :[array](n_ x t_end x l_) residuals

    ## Code

    n_, t_ = X.shape
    k_ = Z.shape[0]
    if isinstance(lam, float) or isinstance(lam, int):
        l_ = 1
        lam = array([lam])
    else:
        l_ = len(lam)

    if smartinverse is None:
        smartinverse = 0

    # if p are not provided, observations are equally weighted
    if p is None:
        p = (1 / t_) @ ones((1, t_))

    # solve optimization
    if l_ == 1 and lam == 0:
        [alpha, beta, s2_U, U] = OrdLeastSquareFPNReg(X, Z, p, smartinverse)
    else:
        # preliminary de-meaning of x and z
        m_X, _ = FPmeancov(X, p)
        m_Z, _ = FPmeancov(Z, p)
        X_c = X - tile(m_X, (1, t_))
        Z_c = Z - tile(m_Z, (1, t_))
        # trick to adapt function solveLasso to the FP framework
        X_p = X_c @ sqrt(np.diagflat(p))
        Z_p = Z_c @ sqrt(np.diagflat(p))
        # initialize variables
        beta = zeros((n_, k_, l_))
        alpha = zeros((n_, l_))
        s2_U = zeros((n_, n_, l_))
        U = zeros((n_, t_, l_))
        # solve lasso
        for l in range(l_):
            for n in range(n_):
                output = solveLasso(X_p[[n], :], Z_p, lam[l], smartinverse)
                beta[n, :, l] = output.beta
            alpha[:, [l]] = m_X - beta[:, :, l] @ m_Z
            U[:, :, l] = X - tile(alpha[:, [l]], (1, t_)) - beta[:, :, l] @ Z
            _, s2_U[:, :, l_ - 1] = FPmeancov(U[:, :, l], p)

    return alpha, beta[..., np.newaxis], s2_U[..., np.newaxis], U


def solveLasso(x, z, lam, smartinverse):
    # ==========================================================================
    #               AUTHOR: GAUTAM V. PENDSE
    #               DATE: 11 March 2011
    # ==========================================================================
    #
    #               PURPOSE:
    #
    #   Algorithm for solving the Lasso problem:
    #
    #           0.5 * (x - beta@z)@(x - beta@z).T + lambda@orbetaor_1
    #
    #   where orbetaor_1 is the L_1 norm i.e., orbetaor_1 = sum(abs((beta)))
    #
    #   We use the method proposed by Fu et. al based on single co-ordinate
    #   descent. For more details see GP's notes or the following paper:
    #
    #   Penalized Regressions: The Bridge Versus the Lasso
    #   Wenjiang J. FU, Journal of Computational and Graphical Statistics,
    #   Volume 7, Number 3, Pages 397?416, 1998
    #
    # ==========================================================================
    #
    #               INPUTS:
    #
    #       =>      x = 1 by t_end response vector
    #
    #       =>      z = k_ by t_end design matrix
    #
    #       => lambda = regularization parameter for L1 penalty
    #
    # ==========================================================================
    #
    #               OPS:
    #
    #       => output.z = supplied design matrix
    #
    #       => output.x = supplied response vector
    #
    #       => output.lambda = supplied regularization parameter for L1 penalty
    #
    #       => output.beta = computed L1 regularized solution
    #
    # ==========================================================================
    #
    #       Copyright 2011 : Gautam V. Pendse
    #
    #               E-mail : gautam.pendse@gmail.com
    #
    #                  URL : http://www.gautampendse.com
    #
    # ==========================================================================

    # ==========================================================================
    #               check input args

    # check size of x
    [n_, t_] = x.shape

    # is x a row vector?
    if (n_ != 1):
        raise ValueError('x must be a 1 by %d vector!!' % t_)
        return []

    # check size of z
    k_, t_1 = z.shape

    # does z have the same number of columns as x?
    if (t_1 != t_):
        raise ValueError('z must have the same number of rows as x!!')
        return []

    # make sure lambda > 0
    if (lam < 0):
        raise ValueError('lam must be >= 0!')
        return []

    # ==========================================================================
    #               initialize the Lasso solution

    # This assumes that the penalty is lambda@beta.T@beta instead of lambda@orbetaor_1
    if smartinverse == 0:
        beta = (x @ z.T).dot(pinv(z @ z.T + 2 * lam))
    else:
        beta = (x @ z.T).dot(SmartInverse(z @ z.T + 2 * lam))

    # ==========================================================================
    #               start while loop

    # convergence flag
    found = 0

    # convergence tolerance
    tol = 1e-6

    while found == 0:

        # save current beta
        beta_old = beta.copy()

        # optimize elements of beta one by one
        for k in range(k_):

            # optimize element i of beta

            # get ith col of z
            z_k = z[[k], :]

            # get residual excluding ith col
            x_k = (x - beta @ z) + beta[0, k] * z_k

            # calulate zi@xi.T and see where it falls
            deltai = z_k @ x_k.T  # 1 by 1 scalar
            if deltai < -lam:
                beta[0, k] = (deltai + lam) / (z_k @ z_k.T)
            elif deltai > lam:
                beta[0, k] = (deltai - lam) / (z_k @ z_k.T)
            else:
                beta[0, k] = 0

        # check difference between beta and beta_old
        if npmax(abs(beta - beta_old)) <= tol:
            found = 1

    # ==========================================================================
    #   save outputs

    output = namedtuple('output', ['z', 'x', 'lam', 'beta'])

    output.z = z
    output.x = x
    output.lam = lam
    output.beta = beta

    return output
