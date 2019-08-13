from numpy import round
from numpy.random import permutation


def RandomSplit(y):
    # This function randomly splits the vector y into two mutually exclusive
    # partitions containing approximately the same number of observations.
    # INPUTS
    #  y  : [vector] (1 x t_end) vector to be splitted
    # OUTPUTS
    #  y1 : [vector] (1 x ~t_end/2) first partition
    #  y2 : [vector] (1 x ~t_end/2) second partition

    ## Code
    t_ = y.shape[1]
    half_t_ = int(round(t_/2))

    # making a random permutation of the given vector y
    y_perm = permutation(y[0]).reshape(1,-1)

    y1 = y_perm[[0],: half_t_]
    y2 = y_perm[[0], half_t_:]
    return y1, y2
