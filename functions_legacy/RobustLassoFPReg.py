import numpy as np
from numpy import ones, zeros, diag, sqrt, r_, atleast_2d
from numpy import sum as npsum
from numpy.linalg import solve, norm
from sklearn.covariance import graph_lasso

from FPmeancov import FPmeancov
from LassoRegFP import LassoRegFP
# from GraphLasso import GraphLasso


def RobustLassoFPReg(X,Z,p,nu,tol,lambda_beta=0,lambda_phi=0,flag_rescale=0):
    # Robust Regression - Max-Likelihood with Flexible Probabilites & Shrinkage
    # (multivariate Student t distribution with given degrees of freedom = nu)
    #  INPUTS
    #   X             : [matrix] (n_ x t_end ) historical series of dependent variables
    #   Z             : [matrix] (k_ x t_end) historical series of independent variables
    #   p             : [vector] flexible probabilities
    #   nu            : [scalar] multivariate Student's t degrees of freedom
    #   tol           : [scalar] or [vector] (3 x 1) tolerance, needed to check convergence
    #   lambda_beta  : [scalar] lasso regression parameter
    #   lambda_phi    : [scalar] graphical lasso parameter
    #   flag_rescale  : [boolean flag] if 0 (default), the series is not rescaled
    #
    #  OPS
    #   alpha_RMLFP   : [vector] (n_ x 1) shifting term
    #   beta_RMLFP    : [matrix] (n_ x k_) optimal loadings
    #   sig2_RMLFP    : [matrix] (n_ x n_) matrix of residuals.T covariances

    # For details on the exercise, see here .

    ## Code
    [n_, t_] = X.shape
    k_ = Z.shape[0]

    # if FP are not provided, observations are equally weighted
    if p is None:
        p = ones((1,t_))/t_
    # adjust tolerance input
    if isinstance(tol, float):
        tol = [tol, tol, tol]

    # rescale variables
    if flag_rescale == 1:
        _,cov_Z=FPmeancov(Z, p)
        sig_Z = sqrt(diag(cov_Z))
        _,cov_X=FPmeancov(X,p)
        sig_X = sqrt(diag(cov_X))
        Z = np.diagflat(1/sig_Z)@Z
        X = np.diagflat(1/sig_X)@X

    # initialize variables
    alpha = zeros((n_,1))
    beta = zeros((n_,k_,1))
    sig2 = zeros((n_,n_,1))

    # 0. Initialize
    alpha[:,[0]], beta[:,:,[0]], sig2[:,:,[0]], U = LassoRegFP(X, Z, p, 0, 0)

    error = ones(3)*10**6
    maxIter = 500
    i = 0
    while any(error>tol) and (i < maxIter):
        i = i+1

        # 1. Update weights
        z2 = np.atleast_2d(U).T@(solve(sig2[:,:,i-1],np.atleast_2d(U)))
        w = (nu+n_)/(nu+diag(z2).T)

        # 2. Update FP
        p_tilde = (p*w) / npsum(p*w)

        # 3. Update output
        # Lasso regression
        new_alpha, new_beta, new_sig2, U = LassoRegFP(X,Z,p_tilde,lambda_beta)
        new_beta = new_beta.reshape(n_,k_,1)
        new_sig2 = new_sig2.reshape(n_,n_,1)
        U = U.squeeze()
        alpha = r_['-1',alpha,new_alpha]
        beta = r_['-1',beta,new_beta]
        sig2 = r_['-1',sig2,new_sig2]
        sig2[:,:,i] = npsum(p*w)*sig2[:,:,i]
        # Graphical lasso
        if lambda_phi != 0:
            sig2[:,:,i],_,_,_=graph_lasso(sig2[:,:,i],lambda_phi)

        # 3. Check convergence
        error[0] = norm(alpha[:,i]-alpha[:,i-1])/norm(alpha[:,i-1])
        error[1] = norm(beta[:,:,i]-beta[:,:,i-1],ord='fro')/norm(beta[:,:,i-1],ord='fro')
        error[2] = norm(sig2[:,:,i]-sig2[:,:,i-1],ord='fro')/norm(sig2[:,:,i-1],ord='fro')

    # Output
    alpha_RMLFP = alpha[:,-1]
    beta_RMLFP = beta[:,:,-1]
    sig2_RMLFP = sig2[:,:,-1]

    # From rescaled variables to non-rescaled variables
    if flag_rescale == 1:
        alpha_RMLFP = diag(sig_X)@alpha_RMLFP
        beta_RMLFP = diag(sig_X)@beta_RMLFP @diag(1/sig_Z)
        sig2_RMLFP = diag(sig_X)@sig2_RMLFP@diag(sig_X).T
    return alpha_RMLFP, beta_RMLFP, sig2_RMLFP
