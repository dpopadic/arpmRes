from numpy import eye
from numpy.linalg import solve


def MinRelEntNormal(mu_pri, s2_pri, v_mu, mu_view, v_sigma, sigma_view):
    # This function computes the Minimum Relative Entropy posterior parameters of the
    # normal distribution which satisfies the constraints due to the views on the expectations
    # and the covariances, and that minimizes the relative entropy with respect to the prior normal distribution of n_
    # arbitrary market variables.
    #  INPUTS
    #   mu_pri     : [vector] (n_ x 1)  prior expectation
    #   s2_pri     : [matrix] (n_ x n_) prior covariance matrix
    #   v_mu       : [matrix] (k_ x n_) matrix specifying the views for the expectations
    #   mu_view    : [vector] (k_ x 1)  vector quantifying the views for the expectations
    #   v_sigma    : [matrix] (s_ x n_) matrix specifying the views for the covariances
    #   sigma_view : [matrix] (s_ x s_) vector quantifying the views for the covariances
    #  OPS
    #   mu_pos     : [vector] (n_ x 1)  posterior expectation
    #   s2_pos     : [matrix] (n_ x n_) posterior covariance matrix

    # For details on the exercise, see here .
    s_, n_ = v_sigma.shape

    mu_pos = mu_pri + s2_pri@v_mu.T@solve(v_mu@s2_pri@v_mu.T,mu_view - v_mu@mu_pri)
    # a = inv(v_sigma@s2_pri@v_sigma.T)
    a = solve(v_sigma@s2_pri@v_sigma.T,eye(s_))
    s2_pos = s2_pri + s2_pri@v_sigma.T@(a@sigma_view@a - a)@v_sigma@s2_pri
    return mu_pos, s2_pos
