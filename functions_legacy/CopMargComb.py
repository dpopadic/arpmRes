import matplotlib.pyplot as plt
from numpy import zeros

from scipy.interpolate import interp1d

plt.style.use('seaborn')


def CopMargComb(x, u, U):
    # This function performs the copula-marginal combination process
    #  INPUTS
    # x : [matrix] (n_ x k_) significant nodes: x(n, k) <= x(n, k+1)
    # u : [matrix] (n_ x k_) cdf grid: u(n, k) = Fn(x(n, k))
    # U : [matrix] (n_ x j_) FP-copula scenarios
    #  OPS
    # X : [matrix] (n_ x j_) FP-joint scenarios

    # For details on the exercise, see here .
    ## Code
    n_, j_ = U.shape

    # joint scenarios by inter-/extra-polation of the grid (u, x)
    X = zeros((n_, j_))

    for n in range(n_):
        interp = interp1d(u[n, :], x[n, :], kind='nearest', fill_value='extrapolate')
        X[n, :] = interp(U[n, :])
    return X