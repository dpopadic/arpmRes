import matplotlib.pyplot as plt
from numpy import arange, exp
from numpy import sum as npsum

plt.style.use('seaborn')

from FPmeancov import FPmeancov


def EwmaFP(epsi, lam):
    # This function computes the exponentially weighted moving average (EWMA)
    # expectations and covariances for time series of invariants
    #  INPUTS
    #   epsi   : [matrix] (n_ x t_end) matrix of invariants observations
    #   lam : [scalar]           half-life parameter
    #  OPS
    #   mu     : [vector] (n_ x 1)  EWMA expectations
    #   sigma2 : [matrix] (n_ x n_) EWMA covariances

    # For details on the exercise, see here .
    ## Code
    _, t_ = epsi.shape
    p = exp(-lam * arange(t_ - 1, 0 + -1, -1)) / npsum(exp(-lam * arange(t_ - 1, 0 + -1, -1)))  # flexible probabilities
    mu, sigma2 = FPmeancov(epsi, p.reshape(1, -1))
    return mu, sigma2
