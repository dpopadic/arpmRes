import matplotlib.pyplot as plt
from numpy import ones, diff, eye

from RobustLassoFPReg import RobustLassoFPReg


def FitVAR1(X, p=None, nu=10**9, lambda_beta=0, lambda_phi=0, flag_rescale=0):
    # This function estimates the 1-step parameters of the VAR[0] process via lasso regression (on first differences)
    #  INPUTS
    #   X                : [matrix] (n_ x t_end) historical series of independent variables
    #   p                : [vector] (1 x t_end) flexible probabilities
    #   nu               : [scalar] degrees of freedom of multivariate Student t
    #   lambda_beta      : [scalar] lasso regression parameter for loadings
    #   lambda_phi       : [scalar] lasso regression parameter for covariance matrix
    #   flag_rescale     : [boolean flag] if 0 (default), the series is not rescaled before estimation

    #  OPS
    #   output1          : [vector](n_ x 1)  output1 = alpha
    #   output2          : [matrix](n_ x n_) output2 = b
    #   output3          : [matrix](n_ x n_) output3 = sig2_U

    ## Code

    dX = diff(X,1,1)
    n_, t_ = dX.shape
    if p is None:
        p = ones((1,t_))/t_

    # robust lasso + glasso regression
    alpha, beta, sig2_U = RobustLassoFPReg(dX, X[:,:-1], p, nu, 10**-6, lambda_beta, lambda_phi, flag_rescale)

    output1 = alpha
    output2 = (eye(n_)+beta)
    output3 = sig2_U

    return output1, output2, output3

