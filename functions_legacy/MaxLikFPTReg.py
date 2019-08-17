import numpy as np
from numpy import r_, array, ones, squeeze
from numpy import sum as npsum
from numpy.linalg import solve, norm

from functions_legacy.OrdLeastSquareFPNReg import OrdLeastSquareFPNReg


def MaxLikFPTReg(X, Z, p, nu, threshold, last=0, smartinverse=0, maxiter=10**5):
    # This function computes the maximum likelihood with Flexible Probabilities
    # (MLFP) estimators of loadings and dispersion of a regression LFM under
    # assumption of t-conditional residuals.
    #  INPUTS
    #   X           :[matrix] (n_ x t_end) time-series of target variables
    #   Z           :[matrix] (k_ x t_end) time-series of factors
    #   p           :[vector] (1 x t_end) flexible probabilities
    #   nu          :[scalar] degrees of freedom
    #   threshold   :[scalar] or [vector](1 x 2) convergence threshold
    #   last        :[scalar]
    #   smartinverse:[scalar] additional parameter: set it to 1 to use
    #                         LRD smart inverse in the regression process
	#   maxiter     :[scalar] maximum number of iterations
    #  OPS
    #   alpha_MLFP  :[tensor] (n_ x iter) shifting parameters computed at each iteration
    #   beta_MLFP   :[tensor] (n_ x k_ x iter) loadings computed at each iteration
    #   s2_MLFP     :[tensor] (n_ x n_ x iter) covariance matrices computed at each iteration
    #   error       :[vector] (2 x 1) relative errors at the last iteration
    # NOTE:
    # if last == 0, all the intermediate steps estimates are returned (DEFAULT)
    # if last!=0 only the final estimates are returned.

    if isinstance(threshold,float):
        threshold = [threshold, threshold, threshold]
    if len(threshold) == 1:
        threshold = [threshold[0],threshold[0],threshold[0]]

    # For details on the exercise, see here .
    ## Code
    n_ = X.shape[0]

    # Initialize

    alpha_MLFP, beta_MLFP, s2_MLFP, U = OrdLeastSquareFPNReg(X, Z, p, smartinverse)

    beta_MLFP = beta_MLFP[...,np.newaxis]
    s2_MLFP = s2_MLFP[...,np.newaxis]

    error = ones(3)*[10**6]
    i = 1
    while any(error > threshold) and i<maxiter:

        # Update weigths
        w_den = nu + npsum(U * (solve(s2_MLFP[:, :, i-1],U)), 0)
        w = (nu + n_) / w_den

        # Update FP
        p_tilde = (p*w) / npsum(p*w)

        # Update output
        new_alpha_MLFP, new_beta_MLFP, new_s2_MLFP, U = OrdLeastSquareFPNReg(X, Z, p_tilde,smartinverse)

        alpha_MLFP = r_['-1',alpha_MLFP, new_alpha_MLFP]
        beta_MLFP = r_['-1',beta_MLFP, new_beta_MLFP[...,np.newaxis]]
        s2_MLFP = r_['-1',s2_MLFP, new_s2_MLFP[...,np.newaxis]]

        s2_MLFP[:, :, i] = npsum(p*w) * s2_MLFP[:, :, i]
        s2_MLFP[:, :, i] = (squeeze(s2_MLFP[:, :, i])+squeeze(s2_MLFP[:, :, i]).T)/2

        # Convergence
        error[0] = norm(alpha_MLFP[:, i] - alpha_MLFP[:, i-1]) / norm(alpha_MLFP[:, i-1])
        error[1] = norm(beta_MLFP[:, :, i] - beta_MLFP[:, :, i-1], ord='fro') / norm(beta_MLFP[:, :, i-1], ord='fro')
        error[2] = norm(s2_MLFP[:, :, i] - s2_MLFP[:, :, i-1], ord='fro') / norm(s2_MLFP[:, :, i-1], ord='fro')
        i = i + 1
    if last != 0:
        alpha_MLFP = alpha_MLFP[:, -1]
        beta_MLFP = beta_MLFP[:, :, -1]
        s2_MLFP = s2_MLFP[:, :, -1]
    return alpha_MLFP, beta_MLFP, s2_MLFP, error
