from numpy import prod, real, sign, diag, eye, abs, log
from numpy import sum as npsum
from numpy.linalg import solve, det, cholesky

from scipy.linalg import lu


def RelEntropyMultivariateNormal(mu, sig2, mu_pri, sig2_pri):
    # This function computes the relative entropy between two normal
    # distributions
    #  INPUTS
    #   mu        [vector]: (n_ x 1) vector of expectations
    #   sig2      [matrix]: (n_ x n_) covariance matrix
    #   mu_pri    [vector]: (n_ x 1) vector of expectations (prior)
    #   sig2_pri  [matrix]: (n_ x n_) covariance matrix (prior)
    #  OPS
    #   obj       [scalar]: entropy

    ## code
    n_ = sig2_pri.shape[0]
    invsig2_pri = solve(sig2_pri,eye(n_))

    mu_diff = (mu - mu_pri)
    obj = 0.5 * mu_diff.T@(invsig2_pri@mu_diff)
    obj = obj + 0.5 * logdet(sig2_pri) - 0.5 * logdet(sig2)
    obj = obj + 0.5 * sum(diag(invsig2_pri@sig2))
    obj = obj - 0.5 * n_

    return obj


def logdet(a):
    # Fast logarithm-determizerost of large matrix

    ## code
    try:
        v = 2 * npsum(log(diag(cholesky(a))))
    except:  ##ok<CTCH>
        dummy, u, p = lu(a)  ##ok<ASGLU>
        du = diag(u)
        c = det(p)*prod(sign(du))
        v = log(c) + npsum(log(abs(du)))
        v = real(v)
    return v
