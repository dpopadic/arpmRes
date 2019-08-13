from numpy import ones, cumsum, diff, tile, r_
from numpy.random import rand
from scipy.stats import t


def PathsCauchy(x0,mu,sigma,tau,j_):
    # This function generates paths for the process x such that the increments
    # dx are iid Cauchy distributed.
    #  INPUTS
    # x0     :[scalar] initial value of process x at time zero
    # mu     :[scalar] location parameter of Cauchy distribution
    # sigma  :[scalar] dispersion arameter of Cauchy distribution
    # tau    :[row vector] vector of times for simulations
    # j_     :[scalar] total number of paths
    #  OPS
    # x      :[matrix](j_ x tau_) array with paths on the rows

    ## Code

    t_ = len(tau)
    r = rand(j_,t_-1)
    dx = t.ppf(r,1,tile(mu*diff(tau,1),(j_,1)),tile(sigma*diff(tau,1),(j_,1)))
    x = r_['-1',x0*ones((j_,1)), x0+cumsum(dx,axis=1)]
    return x
