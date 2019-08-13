import matplotlib.pyplot as plt
import numpy as np
from numpy import sum as npsum
from numpy import zeros, sort, argsort, cumsum, maximum

plt.style.use('seaborn')


def CopMargSep(X, p):
    # This function performs the copula-marginal separation process
    #  INPUTS
    # X : [matrix] (n_ x j_) FP-joint scenarios
    # p : [vector] (1 x j_) Flexible Probabilities
    #  OPS
    # x : [matrix]  (n_ x j_) sorted scenarios: x(n, j) <= x(n, j+1)
    # u : [matrix]  (n_ x j_) sorted cumulative probabilities: u(n, j) <= u(n, j+1)
    # U : [matrix]  (n_ x j_) FP-copula scenarios

    # For details on the exercise, see here .
    ## Preprocess variables
    n_, j_ = X.shape

    if p.shape[0] == n_:
        for n in range(n_):
            p[n] = maximum(p[n], 1 / j_ * 10 ** (-8))
            p[n] = p[n] / npsum(p[n])
    else:
        p = maximum(p, 1 / j_ * 10 ** (-8))
        p = p / npsum(p)

    ## Code
    if j_ <= 10000:
        l = j_ / (j_ + 0.001)  # to be fixed
    else:
        l = j_ / (j_ + 1)

    x, Indx = sort(X, axis=1), argsort(X, axis=1)  # sort the rows of X

    u = zeros((n_, j_))
    U = zeros((n_, j_))

    for n in range(n_):
        I = Indx[n, :]
        cum_p = cumsum(p[0, I])  # cumulative probabilities
        u[n] = cum_p * l  # rescale to be <1 at the far right
        Rnk = np.argsort(I)  # compute ranking
        U[n] = cum_p[Rnk] * l  # grade scenarios

    # clear spurious outputs
    U[U >= 1] = 1 - np.spacing(1)
    U[U <= 0] = np.spacing(1)

    return x, u, U
