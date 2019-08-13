from numpy import arange, trace, prod, array, pi, exp
from numpy.linalg import solve, det

from scipy.special import gamma


def WishartPdf(x,nu,sigma2):
    # This function computes the pdf values from a n_-dimensional Wishart
    # distribution with nu degrees of freedom and scale matrix sigma2
    #  INPUTS
    # x       :[matrix](n_ x n_) coordinates in which the Wishart pdf is evaluated
    # nu      :[scalar] degrees of freedom
    # sigma2  :[matrix](n_ x n_) scale parameter
    #  OPS
    # f       :[scalar] value of Wishart pdf corresponding to coordinates x

    ## Code

    if isinstance(x,float):
        x = array([[x]])
    if isinstance(sigma2,float):
        sigma2 = array([[sigma2]])
    elif sigma2.ndim == 1:
        sigma2 = sigma2.reshape(-1,1)
    n_=x.shape[0]

    #normalization constant
    A=2**((nu*n_)/2)
    B=pi**(n_*(n_-1)/4)
    GAMMA=gamma(arange(nu/2, nu/2-(n_-1)/2-0.5, -0.5))

    K=A*B*prod(GAMMA)

    #pdf
    A1=(1/K)*((det(sigma2))**-(nu/2))
    B1=det(x)**((nu-n_-1)/2)
    C=exp(-0.5*trace(solve(sigma2,x)))

    f=A1*B1*C
    return f
