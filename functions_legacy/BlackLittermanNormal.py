import matplotlib.pyplot as plt
from numpy.linalg import solve

plt.style.use('seaborn')


def BlackLittermanNormal(mu_, s2_, tau, v_mu, mu_view, sig2_view):
    # This function computes the Black-Litterman parameters of the updated
    # normal distribution associated with a normal reference model for the
    # market variables
    #  INPUTS
    #   mu_        : [vector] (n_ x 1) base-case expectation
    #   s2_        : [matrix] (n_ x n_) base-case covariance matrix
    #   tau        : [scalar]  certainty level in the reference model
    #   v_mu       : [matrix] (k_ x n_) pick matrix specifying views
    #   mu_view    : [vector] (k_ x 1) vector quantifying views
    #   sig2_view  : [matrix] (k_ x k_) matrix of confidence level of views
    #  OPS
    #   BLmu       : [vector] (n_ x 1) updated expectation
    #   BLs2       : [matrix] (n_ x n_) updated covariance matrix

    # For details on the exercise, see here .

    ## Code
    pos = (1/tau)*v_mu@s2_@v_mu.T + sig2_view

    BLmu = mu_ + (1/tau)*s2_@v_mu.T@solve(pos,mu_view - v_mu@mu_)
    BLs2 = (1 + (1/tau))*s2_ - (1/tau)**2*s2_@v_mu.T@solve(pos,v_mu@s2_)
    return BLmu, BLs2
