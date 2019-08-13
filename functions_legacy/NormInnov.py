from numpy import ones, sqrt
from numpy.linalg import norm
from scipy.stats import norm


def NormInnov(XZ, m, svec, rho):
    # This function computes the innovation of a bivariate normal
    # variable (X,Z).T
    #  INPUTS
    #   XZ      : [matrix] (2 x j_) joint scenarios of (X,Z).T
    #   m       : [vector] (2 x 1) joint expectation
    #   svec    : [vector] (2 x 1) standard deviations
    #   rho     : [scalar] correlation
    #  OPS
    #   Psi     : [vector] (1 x j_) joint scenarios of innovation

    # For details on the exercise, see here .

    ## Code

    mu = m[0] + rho*(svec[0]/svec[1])*(XZ[1, :] - m[1])
    sigma = sqrt(1 - rho**2) * svec[0]
    j_ = XZ.shape[1]
    Psi = norm.cdf(XZ[0, :], mu, sigma*ones((1, j_)))
    Psi = norm.ppf(Psi, 0, 1)

    return Psi
