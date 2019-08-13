from numpy import trace


def MultivRsquare(s2_U, s2_X, omega2):
    # This function computes the multivariate generalized r-square:
    # R**2{X_orX} = 1 - trace(Cv{X - X_}inv(sig2))/trace((Cv{X}inv(sig2)))
    #  INPUTS
    #   s2_U    : [matrix] (n_ x n_) covariance matrix of U = X - X_
    #   s2_X    : [matrix] (n_ x n_) covariance matrix of X
    #   omega2  : [matrix] (n_ x n_) weighting matrix inv(sig2)
    #  OPS
    #   r_2     : [scalar] r-square

    ## Code
    num = trace(s2_U.dot(omega2))
    den = trace(s2_X.dot(omega2))

    r_2 = 1 - num / den
    return r_2
