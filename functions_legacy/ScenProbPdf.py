from numpy import zeros, eye
from scipy.stats import multivariate_normal


def ScenProbPdf(x_, x, p, h2):
    # This function computes the scenario-probability pdf via Gaussian kernel
    # approximation of the Dirac deltas
    # INPUTS
    #   x_        : [vector] (n_ x 1) points on the selected grid
    #   x         : [matrix] (n_ x j_) scenarios
    #   p         : [vector] (1 x j_) probabilities
    #   h2        : [scalar] bandwidth
    # OUTPUTS
    #   y         : [scalar] Gaussian exponential value

    # For details on the exercise, see here .
    ## Code

    j_ = p.shape[1]

    d = zeros((1, j_))
    for j in range(j_):
        n_ = x_.shape[0]
        d[0,j] = multivariate_normal.pdf(x_.T,x[:, j].T,h2*eye(n_))

    y = p@d.T
    return y
