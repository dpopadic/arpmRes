import matplotlib.pyplot as plt
from numpy import sum as npsum
from numpy import tile
from scipy.stats import gamma

plt.style.use('seaborn')


def Dirichlet(a,numSamples=1):
    #Sample of Dirichlet distribution are obtained drawing gammas
    #see http://en.wikipedia.org/wiki/Dirichlet_distribution#Related_distributions
    #
    # INPUT
    # a          :[vector] (1 x dim) shape parameters for the gamma distributions
    # numSamples :[scalar] number of samples to be generated
    # OP
    # sample     :[matrix] (numSamples x dim) each row contains a sample from a
    #              dim-dimensional Dirichlet distribution with vector of parameters a.
    ##############################################################################

    dim = a.shape[1]
    samples = gamma.rvs(tile(a, (numSamples,1)),1,size=(numSamples,dim))
    samples = samples / tile(npsum(samples,1),(1,dim))

    return samples