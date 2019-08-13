from numpy import zeros, floor

from NormalScenarios import NormalScenarios
from GraphicalLasso import GraphicalLasso


def GraphLasso(s2, lam):
    # This function estimates the covariance and inverse covariance matrix
    # using the graphical lasso algorithm.
    #  INPUTS
    # s2         :[matrix](n_ x n_) initial covariance matrix
    # lambda     :[scalar] penalty for the glasso algorithm
    #  OPS
    # s2_est     :[matrix](n_ x n_) estimated covariance matrix
    # invs2_est  :[matrix](n_ x n_) inverse of the estimated matrix
    # iter       :[scalar] number of performed iterations
    # avgTol     :[scalar] average tolerance of correlation matrix entries before terminating the algorithm
    # hasError   :[flag] flag indicating whether the algorithm terminated erroneously or not

    ## Code

    # generate j_ x n_ sample with target cov = s2 and mean = 0, in order to
    # use the function created by H. Karshenas, which reads data, instead of distributional parameters
    Model = 'Riccati'
    n_ = len((s2))
    j_ = int(floor(n_*2))
    m = zeros((n_,1))
    epsi, _ = NormalScenarios(m,s2,j_,Model)

    s2_est, invs2_est, iter, hasError = GraphicalLasso(epsi.T, lam)
    return s2_est, invs2_est, iter, hasError
