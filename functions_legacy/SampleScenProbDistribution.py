from numpy import cumsum, r_, array, digitize
from numpy.random import rand


def SampleScenProbDistribution(x,p,j_):
    # This function generates a sample from the scenario-probability distribution
    # defined by scenarios x and probabilities p
    # INPUT
    # x  [matrix] (n_ x t_end) scenarios defining the scenario-probability distribution of the random variable X
    # p  [vector] (1 x t_end) probabilities corresponding to the scenarios in x
    # j_ [scalar] number of scenarios to be generated
    # OP
    # X  [matrix] (n_ x t_end) sample from the scenario-probability distribution (x,p)
    #
    # For details on the exercise, see here .

    ## Code
    #empirical cdf
    empirical_cdf=r_['-1',array([[0]]), cumsum(p,axis=1)]

    #create random matrix
    rand_uniform=rand(1,j_)

    # scenarios
    ind = digitize(rand_uniform.flatten(),bins=empirical_cdf.flatten())-1
    X_sample = x[:,ind]
    return X_sample
