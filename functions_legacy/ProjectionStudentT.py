from numpy import arange, cumsum, sqrt
from numpy.fft import fft, ifft
from numpy.linalg import norm
from scipy.stats import norm, t



def ProjectionStudentT(nu, m, s, T):
    ## Perform the horizon projection of a Student t invariant
    #  INPUTS
    #   nu    : [scalar] degree of freedom
    #   s     : [scalar] scatter parameter
    #   m     : [scalar] location parameter
    #   T     : [scalar] multiple of the estimation period to the invesment horizon
    #  OPS
    #   x_Hor : [scalar]
    #   f_Hor : [scalar]
    #   F_Hor : [scalar]

    # set up grid
    N  = 2**14 # coarseness level
    a  = -norm.ppf(10**(-15), 0, sqrt(T))
    h  = 2 * a / N
    Xi = arange(-a+h , a+ h , h ).T

    # discretized initial pdf (standardized)
    f = 1/h*(t.cdf(Xi+h/2,nu) - t.cdf(Xi-h/2,nu))
    f[N-1] = 1/h*(t.cdf(-a+h/2,nu) - t.cdf(-a,nu) + t.cdf(a,nu)-t.cdf(a-h/2,nu))

    # discretized characteristic function
    Phi = fft(f)

    # projection of discretized characteristic function
    Signs = (-1)**(arange(0,N).T*(T-1))
    Phi_T = h**(T-1)*Signs * (Phi**T)

    # horizon discretized pdf (standardized)
    f_T = ifft(Phi_T)

    # horizon discretized pdf and cdf (non-standardized)
    x_Hor = m * T + s * Xi
    f_Hor = f_T / s
    F_Hor = h * cumsum(f_Hor * s)

    return x_Hor, f_Hor, F_Hor
