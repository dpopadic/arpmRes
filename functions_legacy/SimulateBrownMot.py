from numpy import ones, cumsum, diff, sqrt, tile, r_
from numpy.random import randn


def SimulateBrownMot(x0,tau,mu,sigma,j_):
    # This function generates j_ simulated paths of an arithmetic Brownian
    # Motion with drift.
    #  INPUTS
    # x0     :[scalar] starting point
    # tau    :[vector](1 x k_) vector of times
    # mu     :[scalar] drift coefficient
    # sigma  :[scalar] diffusion coefficient
    # j_     :[scalar] total number of simulations
    #  OPS
    # x      :[matrix](j_ x k_) simulated paths

    ## code

    dx = mu*tile(diff(tau),(j_,1))+sigma*tile(sqrt(diff(tau)),(j_,1))*randn(j_,len(tau)-1)
    x = r_['-1',x0*ones((j_,1)), x0+cumsum(dx,axis=1)]
    return x
