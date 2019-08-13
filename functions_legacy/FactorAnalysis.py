import matplotlib.pyplot as plt
import numpy as np
from numpy import sum as npsum
from numpy import zeros, sort, diag, eye, abs, sqrt, tile, maximum, argsort, max as npmax
from numpy.linalg import eig, matrix_rank

plt.style.use('seaborn')

from ConditionalPC import ConditionalPC


def FactorAnalysis(c2,d,k_):
    # This function computes the Low Rank Diagonal Conditional Correlation
    #  INPUT:
    # c2        :[matrix](n_ x n_) correlation matrix
    # d           :[matrix](m_ x n_) matrix of constraints
    # k_          :[scalar] rank of matrix beta.
    #  OP:
    # c2_LRD     :[matrix](n_ x n_) shrunk matrix of the form beta@beta.T+I-diag(beta@beta.T) where beta is a n_ x k_ matrix
    # beta       :[matrix](n_ x k_) low rank matrix: n_ x k_
    # iter        :[scalar] number of iterations
    # constraint  :[scalar] boolean indicator, it is equal to 1 in case the constraint is satisfied, i.e. d@beta = 0

    # For details on the exercise, see here .
    ## Code

    CONDITIONAL=1
    if npsum(abs(d.flatten()))==0:
        CONDITIONAL=0

    n_ = c2.shape[0]

    if k_ > n_-matrix_rank(d):
        raise Warning('k_ has to be <= rho.shape[0]-rank[d]')

    NmaxIter = 1000
    eps1 = 1e-9
    eta = 0.01
    gamma = 0.1
    constraint = 0

    #initialize output
    c2_LRD = c2
    dist = zeros
    iter = 0

    #0. Initialize
    Diag_lambda2, e = eig(c2)
    lambda2 = Diag_lambda2
    lambda2_ord, order = sort(lambda2)[::-1], argsort(lambda2)[::-1]
    lam = np.real(sqrt(lambda2_ord[:k_]))
    e_ord = e[:, order]

    beta = np.real(e_ord[:n_,:k_]@np.diagflat(maximum(lam,eps1)))
    c = c2

    for j in range(NmaxIter):
        #1. Conditional PC
        a = c-eye(n_)+np.diagflat(diag(beta@beta.T))
        if CONDITIONAL==1:
            lambda2, E = ConditionalPC(a, d)
            lambda2 = lambda2[:k_]
            E = E[:,:k_]
            lam = sqrt(lambda2)
        else:
            #if there aren't constraints: standard PC using the covariance matrix
            Diag_lambda2, e = eig(a)
            lambda2 = Diag_lambda2
            lambda2_ord, order = sort(lambda2)[::-1], argsort(lambda2)[::-1]
            e_ord = e[:, order]
            E = e_ord[:,:k_]
            lam = sqrt(lambda2_ord[:k_])

        #2.loadings
        beta_new = E@np.diagflat(maximum(lam,eps1))
        #3. Rows length
        l_n = sqrt(npsum(beta_new**2,1))
        #4. Rows scaling
        beta_new[l_n > 1,:] = beta_new[l_n > 1,:]/tile(l_n[l_n > 1,np.newaxis]*(1+gamma),(1,k_))
        #5. reconstruction
        c = beta_new@beta_new.T+eye(n_,n_)-diag(diag(beta_new@beta_new.T))
        #6. check for convergence
        distance = 1/n_*npsum(sqrt(npsum((beta_new-beta)**2,1)))
        if distance <= eta:
            c2_LRD = c
            dist = distance
            iter = j
            beta = beta_new.copy()
            if d.shape == (1,1):
                tol = npmax(abs(d*beta))
            else:
                tol = npmax(abs(d.dot(beta)))
            if tol < 1e-9:
                constraint = 1
                break
        else:
            beta = beta_new.copy()
            beta = np.real(beta)
            c2_LRD = np.real(c2_LRD)
            c2_LRD = (c2_LRD+c2_LRD.T)/2
    return c2_LRD, beta, dist, iter, constraint

