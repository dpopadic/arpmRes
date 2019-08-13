from numpy import array, zeros, eye
from numpy.linalg import solve
from numpy.random import multivariate_normal as mvnrnd
from scipy.stats import wishart


def RandNormalInverseWishart(mu_0, t_0, sigma2_0, nu_0, j_):
    # Generates a multivariate i.i.d. sample of lenght j_ from the normal-inverse-Wishart distribution:
    #  INPUTS
    #   mu_0      : [vector]
    #   t_0       : [scalar]
    #   sigma2_0  : [matrix]
    #   nu_0      : [scalar]
    #   j_        : [scalar]
    #  OPS
    #   Mu        : [vector]
    #   Sigma2    : [matrix]
    #   InvSigma2 : [matrix]
    #  NOTE
    #   Mu|sigma2   ~ N(mu_0,sigma2/t_0)
    #   inv(Sigma2) ~ W(nu_0,inv(sigma2_0)/nu_0)

    # For details on the exercise, see here .

    ## Code
    if isinstance(mu_0,float):
        n_=1
        mu_0 = array([mu_0])
    else:
        n_ = len(mu_0)

    if sigma2_0.ndim == 1:
        sigma2_0 = sigma2_0.reshape(1,-1)

    invsigma2_0 = solve(sigma2_0,eye(sigma2_0.shape[0]))# inverse of sigma2_0
    phi = (invsigma2_0 / nu_0)[0,0]

    Mu       = zeros((n_,j_))
    Sigma2    = zeros((n_,n_,j_))
    InvSigma2 = zeros((n_,n_,j_))

    for j in range(j_):
        # simulate inv(sigma2)
        InvSigma2[:,:,j] = wishart.rvs(nu_0, phi)
        # compute sigma2
        Sigma2[:,:,j] = solve(InvSigma2[:,:,j],eye(n_))
        # simulate mu
        Mu[0,j] = mvnrnd(mu_0,Sigma2[:,:,j]/t_0)
    return Mu, Sigma2, InvSigma2
