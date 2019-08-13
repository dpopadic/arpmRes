from numpy import ones, std


def ObjectiveTe(data,v_k):
    # Compute the opposite of the tracking error between the benchmark and a
    # replicating portfolio of selected securities
    #  INPUTS
    # data  :[struct] struct with two fields:
    #   1) data.R_b :[vector](1 x t_end) time series of benchmark's linear returns
    #   2) data.R_s :[matrix](n_ x t_end) time series of replicating securities' linear returns
    # v_k   :[vector] indeces of selected replicating securities
    #  OPS
    # mTe   :[scalar] opposite of the tracking error

    # For details on the exercise, see here .

    ## Code
    k = len(v_k)

    R_b = data.R_b
    R_s = data.R_s[v_k,:]

    if not data.w:
        w = ones((k,1))/k
    else:
        w = data.w

    R_w = w.T@R_s# portfolio returns

    R_a = R_w-R_b# active returns

    mTe = -std(R_a,1)
    return mTe
