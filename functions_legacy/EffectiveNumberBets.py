import matplotlib.pyplot as plt
from numpy import log, exp, sum as npsum
from numpy.linalg import solve

plt.style.use('seaborn')


def EffectiveNumberBets(beta, sigma2, a):
    # This function computes the effective number of bets and the
    # diversification distribution according to the Principal Component
    # bets approach or the minimum-torsion approach.
    #  INPUTS
    #   beta     : [vector] 1 x (k_+1)      principal components or minimum-torsion exposures associated to the risk factors Z_0, ..., Z_{k_}
    #   sigma2   : [matrix] (k_+1) x (k_+1) covariance matrix associated to the risk factors Z_0, ..., Z_{k_}
    #   a        : [matrix] (k_+1) x (k_+1) transpose of the matrix whose columns are the orthogonal and normalized eigenvectors of the covariance matrix sigma2
    #                                       (principal components approach)
    #                                       or
    #                                       the minimum-torsion transformation matrix (minimum-torsion approach)
    #  OUTPUTS
    #   enb      : [number] (1 x j_)        minimum-torsion effective number of bets
    #   p        : [vector] (k_+1) x 1      minimum-torsion diversification distribution

    # For details on the exercise, see here .
    ## Code

    p = solve(a.T,beta.T)*(a@sigma2@beta.T)/(beta@sigma2@beta.T)
    p[p==0]=10**(-250)  #avoid log[0-1] in enb computation
    enb = exp(-npsum(p*log(p)))
    return enb, p

