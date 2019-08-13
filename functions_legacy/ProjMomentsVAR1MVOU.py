import numpy as np
from numpy import array, reshape, zeros, eye, atleast_2d
from numpy.linalg import solve

from scipy.linalg import expm, kron


def ProjMomentsVAR1MVOU(x_0, horiz_u, mu, theta, sigma2):
    # Projects conditional first and second moments of VAR1/MVOU process at future horizons
    # model: dXt=-(theta@Xt-mu)dt+sigma@dWt
    #  INPUTS
    #   x_0         [vector]: (n_ x 1) initial conditions
    #   horiz_u     [vector]: (1 x u_)  projection horizons
    #   mu          [vector]: (n_ x 1) drift vector
    #   theta       [matrix]: (n_ x n_) transition matrix
    #   sigma2      [matrix]: (n_ x n_) covariance matrix
    #  OPS
    #   mu_u        [vector]: (n_ x u_) conditional projected long-term means at times horiz_u
    #   sigma2_u    [tensor]: (n_ x n_ x u_) conditional projected covariances at times horiz_u
    #   drift_u     [vector]: (n_ x u_) conditional projected drift at times horiz_u

    ## Code

    n_ = x_0.shape[0]
    if isinstance(horiz_u,float) or isinstance(horiz_u,np.int64):
        u_ = 1
        horiz_u = array([horiz_u])
    else:
        u_ = len(horiz_u)

    mu_u = zeros((n_, u_))
    sigma2_u = zeros((n_, n_, u_))
    drift_u = zeros((n_, u_))

    for u in range(u_):

        # location
        mu_u[:,[u]] = ((eye(n_) - expm(-theta.dot(horiz_u[u])))@(solve(theta,mu))).reshape(-1,1)
        drift_u[:,[u]] = expm(-theta.dot(horiz_u[u]))@x_0 + mu_u[:,[u]]

        # scatter
        thsumth = kron(theta,eye(n_))+kron(eye(n_),theta)
        vecsig2 = reshape(sigma2,(n_**2,1),'F')
        vecsig2_u = solve(thsumth,(eye(n_**2)-expm(-thsumth.dot(horiz_u[u]))))@vecsig2
        sig2_u = reshape(vecsig2_u,(n_,n_),'F')
        sigma2_u[:,:,u] = (sig2_u+sig2_u.T)/2 # makes sigma2_tau numerically symmetric
    return mu_u, atleast_2d(sigma2_u.squeeze()), drift_u
