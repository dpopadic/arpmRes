from numpy import prod, pi, squeeze, diag, exp, sqrt
from numpy.linalg import solve, norm, det
from scipy.stats import norm


def NormalCopulaPdf(u, mu, sigma2):
    # This function computes the pdf of the copula of a multivariate Normal
    # distribution at a generic point u in the unit hypercube
    #  INPUTS
    #   u      : [vector] (n_ x 1) point in the unit hypercube
    #   mu     : [vector] (n_ x 1) vector of expectation
    #   sigma2 : [matrix] (n_ x n_) symmetric and positive covariance matrix
    #  OPS
    #   f_U    : [scalar] pdf of the copula at u

    # For details on the exercise, see here .

    ## Code

    # Compute the inverse marginal cdf's
    sigvec = sqrt(diag(sigma2))
    x = norm.ppf(u.flatten(), mu.flatten(), sigvec).reshape(-1,1)

    # Compute the joint pdf
    n_ = len(u)
    f_X = (2*pi)**(-n_ / 2)*((det(sigma2))**(-.5))*exp(-0.5 * (x - mu).T@(solve(sigma2,(x - mu))))

    # Compute the marginal pdf's
    f_Xn = norm.pdf(x.flatten(), mu.flatten(), sigvec)

    # Compute the pdf of the copula
    f_U = squeeze(f_X/prod(f_Xn))
    return f_U
