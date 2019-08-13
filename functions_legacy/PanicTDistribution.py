from collections import namedtuple

from numpy import ones, zeros, eye, sqrt, tile, r_, array
from scipy.linalg import block_diag
from scipy.stats import chi2, t

from MinRelEntFP import MinRelEntFP
from Tscenarios import Tscenarios
from NormalScenarios import NormalScenarios


def PanicTDistribution(varrho2, r, c, nu, j_):
    # This function generates the joint scenarios and Flexible Probabilities of
    # a Panic t Distribution
    #  INPUTS
    #   varrho2      : [matrix] (n_ x n_) calm market correlation matrix
    #   r            : [scalar] homogeneous panic correlation
    #   c            : [scalar] prob threshold of a high-correlation crash event
    #   nu           : [scalar] degree of freedom
    #   j_           : [scalar] number of scenarios
    #  OPS
    #   X            : [matrix] (n_ x j_) joint scenarios
    #   p_           : [vector] (1 x j_) Flexible Probabilities (posterior via Entropy Pooling)
    #
    #NOTE: Moment matching of t-simulations for nu > 2

    # For details on the exercise, see here .

    ## Code
    n_ = len(varrho2)
    corr_c = varrho2
    corr_p = (1-r)*eye(n_) + r*ones((n_, n_)) # panic corr
    optionT = namedtuple('option', ['dim_red','stoc_rep'])
    optionT.dim_red = 0
    optionT.stoc_rep = 0
    if nu > 1:
        # Calm Component
        Xt_c = Tscenarios(nu, zeros((n_, 1)), corr_c, j_, optionT, 'Riccati')

        # Panic Component
        Xt_p = Tscenarios(nu, zeros((n_, 1)), corr_p, j_, optionT, 'Riccati')
    else:
        s2 = block_diag(corr_c, corr_p)
        Z = NormalScenarios(zeros((2*n_, 1)), s2, j_, 'Riccati')

        # Calm Component
        X_c = Z[:n_,:]
        Chi_2 = chi2.rvs(nu, 1, j_)
        Xt_c = X_c / tile(sqrt(Chi_2 / nu), (n_, 1))

        # Panic Component
        X_p = Z[n_:-1, :]
        Chi_2 = chi2.rvs(nu, 1, j_)
        Xt_p = X_p /tile(sqrt(Chi_2 / nu), (n_, 1))

    # Panic distribution
    B = (Xt_p < t.ppf(c, nu)) # triggers panic
    X = (1-B) * Xt_c + B * Xt_p

    # Perturb probabilities via Fully Flexible Views
    p = ones((1, j_)) / j_ # flat flexible probabilities (prior)
    aeq = r_[ones((1, j_)), X]   # constrain the first moments
    beq = r_[array([[1]]), zeros((n_, 1))]

    p_ = MinRelEntFP(p, None, None, aeq , beq)[0] # compute posterior probabilities
    return X, p_
