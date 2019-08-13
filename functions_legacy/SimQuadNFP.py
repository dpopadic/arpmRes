from numpy import trace, ones, zeros, diag, eye, diagflat
from numpy.linalg import eig, cholesky
from numpy.random import multivariate_normal as mvnrnd

from TwistFPMomMatch import TwistFPMomMatch


def SimQuadNFP(a, b, c, mu, sigma2, j_):
    # This function generates QuadN simulations with Flexible Probabilities
    # whose first moment match the theoretical moment
    #  INPUTS
    #   a      : [scalar] parameter
    #   b      : [vector] (n_ x 1) parameter
    #   c      : [matrix] (n_ x n_) symmetric matrix parameter
    #   mu     : [vector] (n_ x 1) normal expectations
    #   sigma2 : [matrix] (n_ x n_) normal covariances
    #   j_     : [scalar] number of simulations
    #  OPS
    #   Y      : [matrix] (n_ x j_) MC scenarios
    #   p_     : [vector] (1 x j_) twisted flexible probabilities

    # For details on the exercise, see here .

    ## Code

    n_ = mu.shape[0] # number of variables

    # Step 1. Cholesky
    l = cholesky(sigma2)

    # Step 2. Eigen-decomposition
    Diag_lamda, _ = eig(l.T@c@l)

    # Step 3. Change of variables
    gamma = l.T@(b+2*c@mu)

    # Step 4. MC scenarios with Flexible Probabilities
    Z = mvnrnd(zeros(n_), eye((n_)), j_)
    Z = Z.T
    p = ones((1,j_))/j_
    Y = a + b.T@mu + mu.T@c@mu + gamma.T@Z + Diag_lamda.reshape(1,-1)@(Z**2)

    # Step 5. Moment-matching
    mu_ = a + b.T@mu + mu.T@c@mu + trace(c@sigma2)
    p_ = TwistFPMomMatch(Y, p, mu_)
    return Y, p_
