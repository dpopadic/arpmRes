from numpy import zeros, sort, argsort, cumsum, r_

from scipy.integrate import quad


def SpectralIndexSatisf(phi, Pi, h_tilde, p):
    # This function computes a generic spectral index of satisfaction and the corresponding gradient, given
    # the scenario-probability distribution and the spectrum phi.

    # INPUTS:
    # phi       [function]:   spectrum function
    # Pi        [vector]:     n_ x j_ scenarios of the instruments' P&L's
    # h_tilde   [vector]:     n_ x 1 standardized holdings associated to the ex-ante performance Y
    # p         [vector]:     1 x j_ probabilities associated with the scenarios
    # OP:
    # spc       [scalar]: spectral index at the confidence level c corresponding to
    #                     the spectrum function phi
    # grad_spc  [scalar]: gradient of the spectral index at the confidence level c corresponding to
    #                     the spectrum function phi

    # For details on the exercise, see here .

    ## Code
    j_ = p.shape[1]
    Y = h_tilde.T@Pi
    Y_sort, order = sort(Y), argsort(Y)
    p_sort = p[0,order]
    u_sort = r_[0,cumsum(p_sort)]

    w = zeros((1,j_))

    for j in range(j_):
        w[0,j],_ = quad(phi,u_sort[j],u_sort[j+1])

    spc = Y_sort@w.T
    grad_spc = Pi[:,order]@w.T

    return spc, grad_spc
