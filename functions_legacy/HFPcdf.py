from numpy import zeros, sum as npsum

def HFPcdf(x, epsi, p):
    # This def computes the Historical Flexible Probabilities cdf

    # INPUTS
    # x     :[vector] (1 x k_) points
    # epsi  :[vector] (1 x t_end) scenarios
    # p     :[vector] (1 x t_end) Flexible Probabilities

    # OUTPUT
    # F     :[vector] (k_ x 1) cdf values

    k_ = x.shape[1]
    F = zeros((k_, 1))
    for k in range(k_):
        F[k] = npsum(p@(epsi[0] <= x[0, k]))
    return F
