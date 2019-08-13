from numpy import union1d, array, zeros, argmax, setdiff1d


def NaiveSelection(u,data,objective,k_max=None):
    # Naive routine for n_-choose-k selection problems, leading to sub-optimal
    # results.
    #  INPUTS
    # u          :[vector](1 x n_) vector of indices
    # data       :[struct] struct with data needed by the objective function
    # objective  :[handle] handle of the objective function
    # k_max      :[scalar] maximum number of indices to be chosen (optional default = n_)
    #  OPS
    # O_         :[vector](n_ x 1) objective value for each set v_{k}
    # v_         :[cell](n_ x 1) each row contains a set of indices in ascending order of length
    # v_num      :[vector](n_ x 1) cardinality of each set of indices v_[k]

    # For details on the exercise, see here .

    ## Code
    n_ = len(u)
    if k_max is None:
        k_max=n_

    # step 0
    k = 0
    v = array([])
    O_ = zeros((k_max,1))
    v_ = {}
    v_num = zeros((k_max,1))
    while k < k_max:
        h_ = len(u)
        O = zeros((1,h_))
        # step 1
        for h in range(h_):
            O[0,h] = objective(data,u[h:h+1])
        # step 2
        h_star = argmax(O)
        # step 3
        v = union1d(v,u[[h_star]]).astype(int)
        O_[k] = objective(data,v)
        v_[k] = v
        v_num[k] = k+1
        u = setdiff1d(u,u[h_star])
        # step 4
        k = k+1
    return O_,v_,v_num
