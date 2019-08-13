from numpy import sort, argsort, eye, diagflat
from numpy.linalg import eig, solve, cholesky, pinv

from FPmeancov import FPmeancov


def PrincCompFP(X, p, sig2, k_):
    # This function computes the estimators of the shifting term alpha, optimal
    # loadings beta, factor extraction matrix gamma and covariance of residuals
    # s2 for a statistical LFM, by using the non-parametric approach
    #  INPUTS
    #   X         :[matrix] (n_ x t_end) time-series of target variables
    #   p         :[vector] (1 x t_end) flexible probabilities
    #   k_        :[scalar] number of factors
    #  OPS
    #   alpha_PC  :[vector] (n_ x 1) estimator of the shifting term
    #   beta_PC   :[matrix] (n_ x k_) estimator of loadings
    #   gamma_PC  :[matrix] (k_ x n_) estimator of factor-extraction matrix
    #   s2_PC     :[matrix] (n_ x n_) estimator of dispersion of residuals

    ## code
    n_,_ = X.shape
    # compute HFP-expectation and covariance of X
    m_X,s2_X = FPmeancov(X,p)
    # compute the Choleski root of sig2
    sig = cholesky(sig2)
    # perform spectral decomposition
    s2_tmp = solve(sig,(s2_X.dot(pinv(sig))))
    Diag_lambda2, e = eig(s2_tmp)
    lambda2 = Diag_lambda2
    lambda2, index = sort(lambda2)[::-1], argsort(lambda2)[::-1]
    e = e[:,index]
    # compute optimal loadings for PC LFM
    beta_PC = sig@e[:,:k_]
    # compute factor extraction matrix for PC LFM
    gamma_PC = e[:,:k_].T.dot(pinv(sig))
    # compute shifting term for PC LFM
    alpha_PC = (eye(n_)-beta_PC@gamma_PC)@m_X
    # compute the covariance of residuals
    s2_PC = sig@e[:,k_:n_]@diagflat(lambda2[k_:n_])@e[:,k_:n_].T@sig.T
    return alpha_PC, beta_PC, gamma_PC, s2_PC
