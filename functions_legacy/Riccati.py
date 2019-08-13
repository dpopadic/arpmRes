from numpy import zeros, r_, dot
from numpy.linalg import pinv, cholesky
from numpy.linalg.linalg import LinAlgError
from scipy.linalg import schur


def Riccati(phi2, sigma2):
    # This def solves the algebraic Riccati equation s2 = b2 * phi2 * b2.T
    # by means of the Schur decomposition
    #  INPUTS
    #   sigma2 : [matrix] (n_ x n_) symmetric and positive (semi)definite matrix
    #   phi2   : [matrix] (n_ x n_) symmetric and positive (semi)definite matrix
    #  OUTPUTS
    #   b2     : [matrix] (n_ x n_) symmetric matrix positive (semi)definite matrix

    # For details on the exercise, see here .
    ## Code

    n_ = max(sigma2.shape)

    # 1. Block matrix
    h = r_[r_['-1',zeros((n_,n_)), -phi2], r_['-1',-sigma2, zeros((n_,n_))]]

    # 2. Schur decomposition
    try:
        t, u,_ = schur(h, output='real', sort=lambda x: x<0)
    except(LinAlgError):
        t, u = schur(h, output='real')
# 3. Four n_ x n_ partitions
    u_oneone = u[:n_,:n_]
    u_twoone = u[n_:, :n_]

    # Output
    b2 = dot(u_twoone, pinv(u_oneone))
    return b2
