import matplotlib.pyplot as plt
import numpy as np
from numpy import max as npmax
from numpy import zeros, where, mod, ceil

from scipy.misc import comb

plt.style.use('seaborn')


def BackwardSelection(v,data,objective,step=1,k_min=1):
    # Backward stepwise routine for n_-choose-k selection problems
    #  INPUTS
    # v          :[vector](1 x n_) vector of indices
    # data       :[struct] struct with data needed by the objective function
    # objective  :[handle] handle of the objective function
    # step       :[scalar] number of indices to be added at each iteration
    # k_min      :[scalar] minimum number of indices to be chosen (optional default = 1)
    #  OPS
    # O_         :[vector](k_ x 1) objective value for each set v_{k}
    # v_         :[cell](k_ x 1) each row contains a set of indices in ascending order of length
    # v_num      :[vector](k_ x 1) cardinality of each set of indices v_[k]

    # For details on the exercise, see here .

    ## Code

    n_ = len(v)

    # step 0
    k_ = int(ceil(n_/step))
    k = k_-1
    k_stop=int(ceil(k_min/step)-mod(k_min,step))-1

    O_ = zeros((k_,1))
    O_[k] = objective(data,v)
    v_ = {}
    v_[k] = v
    v_num = zeros((k_,1))
    v_num[k] = n_
    while k > k_stop:
        n = comb(v,step)
        if n.size==0:
            n = v
        h_ = n.shape[0]
        O_k = zeros((1,h_))
        for h in range(h_):
            # step 1
            v_h = np.setdiff1d(v,n[h])
            # step 2
            O_k[0,h] = objective(data,v_h)
        # step 3
        O_star, h_star = npmax(O_k[0]), np.argmax(O_k[0])
        O_[k-1] = O_star
        # step 4
        v = np.setdiff1d(v,n[h_star])
        v_[k-1] = v
        v_num[k-1] = len(v)
        # step 5
        k = k-1

    O_ = where(v_num==0,0,O_)
    v_ = where(v_num==0,0,v_)
    v_num = where(v_num==0,0,v_num)
    return O_, v_, v_num

