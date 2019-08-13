from numpy import ones, eye, zeros, r_
from numpy.random import multivariate_normal as mvnrnd

from TwistScenMomMatch import TwistScenMomMatch


def NormalScenarios(mu_, sigma2_, j_, method='Riccati', d=None):
    # This def generates antithetic normal simulations whose
    # moments match the theoretical moments
    #  INPUTS
    #   mu_     : [vector] (n_ x 1) vector of means
    #   sigma2_ : [matrix] (n_ x n_) dispersion matrix
    #   j_      : [scalar] (even) number of simulations
    #   method  : [string] Riccati (default), CPCA, PCA, Cholesky-LDL, Gram-Schmidt
    #   d       : [matrix] (k_ x n_) full rank constraints matrix for CPCA
    #  OUTPUTS
    #   X_      : [matrix] (n_ x j_) normal matrix of simulations
    #   p       : [vector] (1 x j_) vector of Flexible Probabilities
    # NOTE: Use always a large number of simulations j_ >> n_ to ensure that
    # NormalScenarios works properly

    # For details on the exercise, see here .
    ## Code

    n_ = max(mu_.shape)     # number of variables

    # Step 1. normal MC scenarios
    n_scenarios = int(j_/2)
    X_tilde = mvnrnd(zeros(n_), eye(n_), n_scenarios)
    X_tilde = X_tilde.T

    # Step 2. Anthitetic (mean = 0)
    X = r_['-1',X_tilde, -X_tilde]
    p = ones((1, j_))/j_ # flat probabilities

    # Step 3. Twisted t MC scenarios
    X_ = TwistScenMomMatch(X, p, mu_, sigma2_, method, d)
    return X_, p
