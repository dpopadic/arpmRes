from numpy import sum as npsum
from numpy import zeros, abs, mean

from CopMargSep import CopMargSep


def SWDepMeasure(X, probs):
    # This function estimates the Schweizer and Wolff measure of dependence
    # between two random variables by means of Monte Carlo simulations
    #  INPUTS
    #   X     : [matrix]  (2 x j_) joint scenarios
    #   probs : [vector]  (1 x j_) vector of Flexible probabilities
    #  OPS
    #   dep   : [scalar]  Schweizer-Wolff measure estimate

    # For details on the exercise, see here .

    ## Code
    _, _, U = CopMargSep(X, probs) # grades scenarios

    j_ = X.shape[1] # number of scenarios

    g = zeros((j_, j_))
    for i in range(j_):
        for k in range(j_):
            g[i, k] = abs(npsum(probs*(U[0] <= i/j_)*(U[1] <= k/j_))-(i*k)/j_**2)

    dep = 12 *mean(g.flatten())
    return dep
