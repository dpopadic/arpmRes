import numpy as np

from numpy import ones, zeros
from numpy.linalg import norm
from scipy.linalg import expm, logm
# from cvxpy import Variable, Minimize, Problem, normNuc, upper_tri
from cvxpy import *


def ProjMarkovChain(p, tau):
    # Project the Markov chain p to horizon tau
    #  INPUTS
    #   p       : [matrix] (n_ x n_) Markov chain matrix
    #   tau     : [scalar] projection horizon
    #  OPS
    #   p_tau   : [matrix] (n_ x n_) projected Markov chain matrix
    #   g       : [matrix] projected Markov chain generator

    ## Code

    # transition probability matrix dimension
    n_ = len(p)

    # one-step generator
    l = logm(p)

    # QP routine
    g =Variable(n_,n_)
    objective = Minimize(norm(g - l))
    constraints = [
        g*ones((n_,1)) == zeros((n_,1))
    ]
    for i in range(1,n_):
        for j in range(0,i):
            constraints.append(g[i,j]>=0)
        for j in range(i,n_):
            constraints.append(g[i-1,j]>=0)

    prob = Problem(objective,constraints)
    prob.solve(verbose=False)
    # projection
    g = np.array(g.value)
    p_tau = expm(tau*g)

    return p_tau, g
