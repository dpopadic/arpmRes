import matplotlib.pyplot as plt
from numpy import arange
from numpy.linalg import norm
from scipy.stats import norm

plt.style.use('seaborn')


def CornishFisher(mu, sd, sk, c=None):
    # This function computes the Cornish-Fisher approximation (up to the second term)
    # of the quantile function of a generic random variable, given its mean,
    # standard deviation and skweness for an arbitrary set of confidence
    # levels.
    #  INPUTS
    #  mu      : [scalar] mean
    #  sd      : [scalar] standard deviation
    #  sk      : [scalar] skewness
    #  c       : [vector] (arbitrary length) confidence levels
    #  OP
    #  q       : [scalar] Cornish-Fisher approx

    # For details on the exercise, see here .

    if c is None:
        c = arange(.001,1,0.001)
    z = norm.ppf(c)

    # Cornish-Fisher expansion
    q = mu + sd@(z + sk / 6 * (z**2 - 1))
    return q

