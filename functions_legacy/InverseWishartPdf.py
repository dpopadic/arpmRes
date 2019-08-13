from numpy import prod, arange, trace, array, pi, exp
from numpy.linalg import det

from scipy.special import gamma


def InverseWishartPdf(x,nu,psi):
    # This function evaluates the pdf of an inverse-Wishart distribution, with
    # nu degrees of freedom and scale matrix psi, at point x.
    #  INPUTS
    # x    :[matrix](n_ x n_) point at which the pdf is evaluated
    # nu   :[scalar] degrees of freedom
    # psi  :[matrix](n_ x n_) scale parameter of the inverse-Wishart distribution
    #  OPS
    # f    :[scalar] value of the inverse-Wishart pdf

    ## Code
    if isinstance(x,float):
        x = array([[x]])
    elif x.ndim == 1:
        x = x.reshape(-1,1)
    if isinstance(psi,float) or  isinstance(psi,int):
        psi = array([[psi]])
    elif psi.ndim == 1:
        psi = psi.reshape(-1,1)
    n_=x.shape[0]

    #normalization constant
    a=2**((nu*n_)/2)
    b=pi**(n_*(n_-1)/4)
    Gamma=gamma(arange(nu/2,nu/2-(n_-1)/2-0.5,-0.5))

    k=a*b*prod(Gamma)

    #pdf
    a1=(1/k)*(det(psi)**(nu/2))
    b1=(det(x))**(-(nu+n_+1)/2)
    c=exp(-0.5*trace(psi/x))

    f=a1*b1*c

    return f
