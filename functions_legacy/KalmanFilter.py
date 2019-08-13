import numpy as np
from numpy import eye, r_
from numpy.linalg import pinv


def KalmanFilter(X,alpha,beta,s2,alpha_z,beta_z,s2_z,option=None):
    # This function extracts the time series of hidden factors Z from the time
    # series of observed states X, basing on the following Kalman filter model:
    #
    # X_t = alpha + beta@Z_t + U_t  U_t ~ N(0,s2) (observation equation)
    # Z_t = alpha_z + beta_z@Z_{t-1} + U_{z,t}  U_{z,t} ~ N(0,s2_z) (transition equation)
    #
    #  INPUTS
    # X        :[matrix](n_ x t_end) time series of observed variables
    # alpha    :[vector](n_ x 1) shift parameter of observation equation
    # beta     :[matrix](n_ x k_) parameter of observation equation
    # s2       :[matrix](n_ x n_) covariance of residuals U_t
    # alpha_z  :[vector](k_ x 1) shift parameter of transition equation
    # beta_z   :[matrix](k_ x k_) parameter of transition equation
    # s2_z     :[matrix](k_ x k_) covariance of residuals U_{z,t}
    # option   :[struct] structure specifying the initial values of hidden factors Z and their uncertainty.
    #                    In particular:
    #                    option.Z   :[vector](k_ x 1) initial value of hidden factor
    #                    option.p2  :[matrix](k_ x k_) uncertainty on the initial value Z of hidden factors
    #  OPS
    # Z        :[matrix](k_ x t_end) time series of hidden factors

    ## Code

    _,t_ = X.shape
    k_ = len(alpha_z)

    if option is None:
    # set the initial value Z[:,0] for the hidden factors
        Z = beta.T@X[:,[0]]
    # set the initial covariance of Z[:,0], expressing the uncertainty of measurement
        p2 = 10*eye(k_).reshape(k_,k_,1)
    else:
        Z = option.Z
        p2 = option.p2

    for t in range(1,t_):
        # estimate step:
        Z_tilde = alpha_z + beta_z@Z[:,[t-1]]
        p2_tilde = beta_z@p2[:,:,t-1]@beta_z.T + s2_z

        # correction step:
        u = X[:,[t]] - alpha - beta@Z_tilde
        s2_u = s2 + beta@p2_tilde@beta.T
        kappa = p2_tilde@beta.T.dot(pinv(s2_u))
        Z = r_['-1',Z, Z_tilde + kappa@u]
        p2 = r_['-1',p2, ((eye(k_) - kappa@beta)@p2_tilde)[...,np.newaxis]]
    return Z
