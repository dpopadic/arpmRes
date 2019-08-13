from numpy import zeros, where, r_


def MatchTime(x, t, t_new):
    # This fuction computes a sub-vector x_new of vector x, containing the
    # entries for which t_new matches t.
    # INPUTS
    #  x      :[vector](1 x t_end) vector of data corresponding to points in time within t
    #  t      :[vector](1 x t_end) vector of times corresponding to the entries of x
    #  t_new  :[row vector] vector of new times
    # OP
    #  x_new  :[vector](1 x k_) vector of data equal to x when t = t_new
    #  t_new  :[vector](1 x k_) updated t_new vector

    j = 0
    x_new = zeros(1)
    for k in range(len(t_new[0])):
        index = where(t[0]==t_new[0,j])[0]
        if index.size>0:
            x_new = r_[x_new, x[[0],index]]
            j = j+1
        else:
            t_new = delete(t_new, j, 1)
    x_new = x_new[1:]
    return x_new, t_new
