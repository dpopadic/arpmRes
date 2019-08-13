import matplotlib.pyplot as plt
import numpy as np
from numpy import max as npmax
from numpy import union1d, zeros, ceil

from scipy.misc import comb

plt.style.use('seaborn')


def ForwardSelection(u,data,objective,step=1,k_max=None):
    # Forward stepwise routine for n_-choose-k selection problems
    #  INPUTS
    # u          :[vector](1 x n_) vector of indices
    # data       :[struct] struct with data needed by the objective function
    # objective  :[handle] handle of the objective function
    # step       :[scalar] number of indices to be added at each iteration
    # k_max      :[scalar] maximum number of indices to be chosen (optional default = n_)
    #  OPS
    # O_         :[vector](k_ x 1) objective values for each set v_{k}
    # v_         :[cell](k_ x 1) each row contains a set of indices in ascending order of length
    # v_num      :[vector](k_ x 1) cardinality of each set of indices v_[k]

    # For details on the exercise, see here .

    ## Code

    n_ = max(u.shape)
    if k_max is None:
        k_max = n_

    # step 0
    v = []
    k = 0
    k_ = int(ceil(k_max/step))
    O_ = zeros((k_,1))
    v_ = {}
    v_num = zeros((k_,1))
    while k < k_:
        n = comb(u,step)
        if n.size ==0:
            n = u
        h_ = n.shape[0]
        O_k = zeros((1,h_))
        for h in range(h_):
            # step 1
            v_h = union1d(v,n[[h]]).astype(int)
            # step 2
            O_k[0,h] = objective(data,v_h)

        # step 3
        O_star, h_star = npmax(O_k), np.argmax(O_k)
        O_[k] = O_star
        # step 4
        v = union1d(v,n[[h_star]]).astype(int)
        v_[k] = v
        v_num[k] = len(v)
        u = np.setdiff1d(u,n[h_star])
        # step 5
        k = k+1
    return O_, v_, v_num

