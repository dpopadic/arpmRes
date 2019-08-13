from numpy import diagflat, zeros, ones
from numpy.linalg import svd, solve
import numpy as np
from cvxopt import matrix
from cvxopt.solvers import qp


def SDFkern(V,v_tnow,p):
    # This function computes the Stochastic Discount Factor which is obtained
    # as the smallest perturbation of the projection SDF which makes it strictly positive
    #INPUTS
    #  V          : [matrix] n_ x j_ payoff matrix
    #  v_tnow     : [vector] n_ x 1  vector of curret values
    #  p          : [vector] 1 x j_  row vector of probabilities
    #OP
    #  SDF_Kern   : [vector] 1 x j_ row vector of SDF scenarios

    # For details on the exercise, see here .

    ## Code
    #parameters
    n_,j_ = V.shape

    # Compute the projection SDF_proj
    SDF_proj = V.T@solve(V@diagflat(p)@V.T,v_tnow)

    # Get a kernel basis via singular value decomposition
    _,_,B = svd(V@diagflat(p))
    B = B[:,n_:j_]

    # Minimization problem
    results = qp(matrix(B.T@B),matrix(zeros((j_-n_,1))),matrix(-B),matrix(SDF_proj))
    x_kern = np.matrix(results['x'])

    #Generate the output
    SDF_kern = SDF_proj+B@x_kern
    SDF_kern=SDF_kern.T

    return SDF_kern
