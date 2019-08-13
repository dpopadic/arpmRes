from numpy import zeros, floor, diag, sqrt

from NormalScenarios import NormalScenarios
from SMTCovarEst import SMTCovarEst


def SMTCovariance(sigma2,k):
    # This function computes the sparse matrix transformation estimate for the
    # covariance matrix sigma2 by using the number of rotations indicated in
    # vector k
    #  INPUTS
    # sigma2      :[matrix](n_ x n_) starting covariance matrix
    # k         :[vector](1 x k_) vector containing the number of sparse rotations to be used
    #  OPS
    # sigma2_SMT  :[matrix](n_ x n_) transformed covariance matrix

    n_ = sigma2.shape[0]

    #generate (j_ x n_) sample with target cov = sigma2  and mean = 0
    Model = 'Riccati'
    j_ = floor(n_*2)
    m = zeros((n_,1))
    epsi = NormalScenarios(m,sigma2,j_,Model)

    sigma2_SMT = zeros((n_,n_,len(k)))

    for i in range(len(k)):
        e,lam,ArraySMT=SMTCovarEst(epsi,k[i])
        CovSMT = e@lam@e.T
        sigma2_SMT[:,:,i] = CovSMT/(diag(sqrt(CovSMT))@diag(sqrt(CovSMT)).T)
    return sigma2_SMT,ArraySMT
