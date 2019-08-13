import numpy as np
from numpy import zeros
from numpy.random import rand, randn


def MetrHastAlgo(f_hat,f_pri,theta_0,j_):
    # This function performs the Metropolis-Hastings algorithm to generate a
    # sample of length j_ from the univariate posterior distribution defined by
    # the conditional likelihood f_hat and the prior f_pi. The candidate-
    # generating function is a normal pdf with unitary variance, parametrized
    # by the expectation.
    #  INPUTS
    # f_hat    :[handle] handle of conditional likelihood pdf
    # f_pri    :[handle] handle of prior pdf
    # theta_0  :[scalar] initial value
    # j_       :[scalar] length of the sample to be generated
    #  OPS
    # theta    :[vector](1 x j_) generated sample
    # a_rate   :[scalar] acceptance rate

    # For details on the exercise, see here .

    ## code
    u = rand(1,j_)# pre-allocate the draws from the uniform distribution (step 2)
    gamma = randn(1,j_)# pre-allocate the draws from standard normal distribution

    theta = zeros(j_+1)
    theta[0] = theta_0
    accepted = 0
    for j in range(j_):
        # step 1
        xi = theta[j] + gamma[0,j]# generate candidate from a distribution N(theta([j],1))
        # step 2 already performed
        # step 3
        beta = (f_hat(xi)*f_pri(xi))/(f_hat(theta[j])*f_pri(theta[j]))
        alpha = np.minimum(beta, 1)
        # step 4
        if u[0,j] <= alpha:
            theta[j+1]= xi
            accepted = accepted+1
        else:
            theta[j+1]= theta[j]
    theta = theta[1:]
    a_rate = accepted/j_
    return theta,a_rate
