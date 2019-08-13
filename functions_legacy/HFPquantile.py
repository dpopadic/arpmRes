from numpy import ones, zeros, sort, argsort


def HFPquantile(x,conf,p=None):
    # This function computes the quantile of a Flexible probabilities
    # distribution
    #  INPUTS
    #  x            :[vector](1 x t_end) scenarios
    #  conf         :[vector](1 x n_ql) confidence levels
    #  p  (optional):[vector](1 x t_end) Flexible Probabilities
    #  OPS
    #  q_HFP        :[vector](1 x n_ql) quantiles

    ## Code

    n_ql = conf.shape[1]
    # if the third argument is missing, the Flexible Probabilities are set to be uniform
    if p is None:
        p = (1/x.shape[1])*ones((1,x.shape[1]))

    x_sort, y = sort(x), argsort(x)

    p_sort = p[0,y]

    q_HFP = zeros((1,n_ql))
    cum = 0
    j = 0
    t = 0
    while j < n_ql:
        while cum < conf[0,j] and t < x.shape[1]:
            cum = cum + p_sort[0,t]
            t = t+1
        if t == 0:
            q_HFP[0,j] = x_sort[0,0]
        else:
            q_HFP[0,j] = x_sort[0,t-1]
        j = j+1
    return q_HFP