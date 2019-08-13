
from numpy import tile


def FPmeancov(x, p):
    '''This def computes the mean and covariance matrix of a Flexible Probabilities distribution
    INPUT
    x    :[matrix] (i_ x t_end) scenarios
    p    :[vector] ( 1 x t_end) Flexible Probabilities
    OUTPUT
    m    :[vector] (i_ x 1)  mean
    s2   :[matrix] (i_ x i_) covariance matrix'''

    if p.shape[1] == 1:
        p = p.T

    i_, t_ = x.shape
    m = x.dot(p.T)   # mean
    X_cent = x - tile(m, (1, t_))
    s2 = (X_cent*tile(p, (i_, 1))).dot(X_cent.T) # covariance matrix
    s2 = (s2 + s2.T)/2 # eliminate numerical error and make covariance symmetric

    return m, s2
