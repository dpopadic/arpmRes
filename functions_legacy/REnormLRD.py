from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from numpy import trace, reshape, zeros, diag, eye, log, r_, diagflat, kron, array
from numpy.linalg import solve, det

plt.style.use('seaborn')
np.seterr(invalid='ignore')


def REnormLRD(theta, mu_, invs2_, n_, k_, matrix=None):
    # This function computes the relative entropy with low-rank-diagonal
    # structure on covariance
    #  INPUTS
    # theta   : [vector] (n_@(2 + k_) x 1) vector of variables: theta = (mu, beta.flatten(), d)
    # mu_     : [vector] (n_ x 1) base-case expectation
    # invs2_  : [vector] (n_ x n_) inverse base-case covariance
    # n_      : [scalar] market dimension
    # k_      : [scalar] number of factors
    # matrix  : [struct] constant matrices for derivatives (optional)
    #  OPS
    # obj     : [scalar] entropy value
    # grad    : [vector] (n_@(2 + k_) x 1) gradient
    # hess    : [matrix] (n_@(2 + k_) x n_@(2 + k_)) hessian

    # For details on the exercise, see here .

    if matrix is None:
        matrix = namedtuple('matrix', ['hm1', 'km'])
        matrix.hm1 = array([])
        matrix.km = array([])

    ## Code
    mu, s2, beta, d = theta2param(theta, n_, k_) # set mu, b, d parameters
    # relative entropy
    obj = 0.5 * (trace(s2@invs2_) - log(det(s2@invs2_)) + (mu - mu_).T@invs2_ @(mu - mu_) - n_)

    # inverse of s2 using binomial inverse theorem
    d2 = d ** 2
    diag_ = diagflat(1/d2)
    tmp = solve(beta.T@diag_@beta + eye(k_),beta.T@diag_)
    invs2 = diag_ - (diag_@beta@tmp)

    # gradient
    grad_mu = invs2_@(mu - mu_)
    v = (invs2_ - invs2)
    q = v@beta
    grad_b = q.reshape((-1,1),order='F')
    grad_d = diag(v).reshape((-1,1),order='F') * d
    grad = r_[grad_mu, grad_b, grad_d]

    i_k = eye(k_)
    i_n = eye(n_)

    if matrix.hm1.size==0:
        matrix.hm1 = zeros((n_**2, n_))
        for n in range(n_):
            matrix.hm1 = matrix.hm1 + kron(i_n[:, [n]], diagflat(i_n[:, [n]]))
    if matrix.km.size==0:
        matrix.km = zeros((k_*n_, k_*n_))
        for k in range(k_):
            matrix.km = matrix.km + kron(kron(i_k[:, [k]], i_n),i_k[:, [k]].T)

    # hessian
    Diag_d = diagflat(d)
    grad2_mumu = invs2_
    grad2_dd = (2 * Diag_d@invs2) * (invs2@Diag_d) + diagflat(diag(v))
    grad2_bd = kron(beta.T@invs2, invs2@Diag_d)*2@matrix.hm1
    grad2_bb = kron(beta.T@invs2@ beta, invs2) + matrix.km@kron(invs2@beta, beta.T@invs2) + kron(i_k, v)

    hess = r_[r_['-1',grad2_mumu,          zeros((n_, n_*k_)),   zeros((n_, n_))],
              r_['-1',zeros((n_*k_, n_)),  grad2_bb,             grad2_bd],
              r_['-1',zeros((n_, n_)),     grad2_bd.T,           grad2_dd]]

    return obj, grad, hess


def theta2param(theta, n_, k_):
    id = range(n_)
    mu = reshape(theta[id], (-1, 1),'F')
    id = range(n_, n_ + n_*k_)
    b  = reshape(theta[id], (n_, k_),'F')
    id = range(n_+ n_*k_, n_*(2 + k_))
    d = reshape(theta[id], (-1, 1),'F')

    s2 = b@b.T + diagflat(d**2)
    return mu, s2, b, d
