from numpy import zeros

from ProjMarkovChain import ProjMarkovChain


def ProjectTransitionMatrix(p, tau):
    #This function performes the projection of the transition matrix p to the
    #horizon tau
    #  INPUTS
    #   p       : [matrix] (n_ x n_) transition matrix
    #   tau     : [scalar] projection horizon
    #  OPS
    #   p_tau   : [matrix] (n_ x n_) projected transition matrix

    p_tau = ProjMarkovChain(p,tau)[0]
    p_tau[-1,:] = zeros((1,p.shape[1]))
    p_tau[-1,-1] = 1
    return p_tau
