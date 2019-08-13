from numpy import sum as npsum
from numpy import zeros, sort, argsort, cumsum, r_
from scipy.stats import norm


def SmoothQuantile(c, X, p):
    # This function computes the smooth quantile at a given confidence level
    # associated to a scenario-probability distribution of a random variable X.
    # INPUTS
    # c [scalar]:       confidence level
    # X [vector]:       n_ x j_ scenarios of the random variable X
    # p [vector]:       1 x j_ probabilities associated with the scenarios
    # OP
    # q [scalar]:       smooth quantile at the confidence level c
    # w [vector]:       n_x1 weights of the smooth quantile
    # order [vector]:   vector of indices of sorted scenarios

    # For details on the exercise, see here .

    ## Code
    j_ = p.shape[1]

    X_sort, order = sort(X), argsort(X)
    p_sort = p[0,order]
    u_sort = r_[0,cumsum(p_sort)]

    h = 0.25*(j_**(-0.2))
    w = zeros((1,j_))

    for j in range(j_):
        w[0,j] = norm.cdf(u_sort[j+1],c,h) - norm.cdf(u_sort[j],c,h)
    w=w/npsum(w)

    q = X_sort@w.T

    return q, w, order
