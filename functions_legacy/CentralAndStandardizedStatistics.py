import matplotlib.pyplot as plt
from numpy import ones, zeros, sqrt, tile

plt.style.use('seaborn')


def CentralAndStandardizedStatistics(k_, x, p=None):
    ## Compute central and standardized statistics
    #  INPUTS
    # k_        :[scalar] highest degree for the central moment
    # x         :[vector](1 x t_end) draws from the distribution
    # p         :[vector](1 x t_end) flexible probabilities associated with the scenarios x
    #                             default: all equal probabilities
    #  OPS
    # gamma     :[vector](1 x k_) standardized statistics up to order k_
    # mu_tilde  :[vector](1 x k_) central moments up to order k_

    ## Code
    t_ = x.shape[1]
    if p is None:
        p = ones((1,t_))/t_

    # compute central moments
    mu_tilde = zeros((1, k_))

    mu_tilde[0,0] = x@p.T
    if k_>1:
        x_cent = x-tile(mu_tilde[0,[0]],(1,t_))
        for k in range(1,k_):
            mu_tilde[0,k] = (x_cent**(k+1))@p.T

    # compute standardized statistics
    gamma = mu_tilde.copy()
    if k_>1:
        gamma[0,1] = sqrt(mu_tilde[0,1])
    if k_>2:
        for k in range(2,k_):
            gamma[0,k] = mu_tilde[0,k] / (gamma[0,1]**(k+1))
    if k_>3:
        gamma[0,3] = gamma[0,3]-3# excess kurtosis
    return gamma, mu_tilde
