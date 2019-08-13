from numpy import prod, pi, squeeze, diag, sqrt
from numpy.linalg import solve, det
from scipy.special import gamma
from scipy.stats import t


def StudentTCopulaPdf(u, nu, mu, sigma2):
    # This function computes the pdf of the copula of a multivariate Student t
    # distribution at a generic point u in the unit hypercube
    #  INPUTS
    #   u      : [vector] (n_ x 1) point in the unit hypercube
    #   nu     : [scalar] degrees of freedom
    #   mu     : [vector] (n_ x 1) vector of location
    #   sigma2 : [matrix] (n_ x n_) symmetric and positive scatter matrix
    #  OPS
    #   f_U    : [scalar] pdf of the copula at u

    # For details on the exercise, see here .

    ## Code

    # Compute the inverse marginal cdf's
    sigvec = sqrt(diag(sigma2)).reshape(-1,1)
    x = mu + sigvec * t.ppf(u, nu)

    # Compute the joint pdf
    n_ = len(u)
    z2 = (x - mu).T@solve(sigma2,(x - mu))
    const  = (nu*pi)**(-n_ / 2)*gamma((nu + n_) / 2) / gamma(nu / 2)*((det(sigma2))**(-.5))
    f_X = const*(1 + z2 / nu) ** (-(nu + n_) / 2)

    # Compute the marginal pdf's
    f_Xn = t.pdf((x - mu) / sigvec, nu) / sigvec

    # Compute the pdf of the copula
    f_U = f_X / prod(f_Xn)
    return squeeze(f_U)
