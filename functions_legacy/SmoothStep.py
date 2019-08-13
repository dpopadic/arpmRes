from numpy.linalg import norm
from scipy.stats import norm


def SmoothStep(x,h):
    # This function computes the step function smoothed via Gaussian cdf
    # INPUTS
    #   x         : [matrix] (n_ x n_) grid points
    #   h         : [scalar] smoothing parameter
    # OUTPUTS
    #   y         : [matrix] (n_ x n_) smoothed values ad grid points

    ## Code
    y = norm.cdf(x,0,h)
    return y
