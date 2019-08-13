import matplotlib.pyplot as plt
from numpy import arange, ones, zeros
from numpy import sum as npsum

from scipy.optimize import least_squares
from scipy.special import gamma

from autocorrelation import autocorrelation


def FitFractionalIntegration(dx, l_, d0):
    # Fit of a fractional integration process on X
    #  INPUTS
    #  x         : [vector] (1 x t_end) data dx=diff[x]
    #  l_        : [scalar] fractional integration process is approximated considering the first l_ terms of its Taylor expansion
    #  d0        : [scalar] initial guess for the parameter d
    # OUTPUTS
    # d          : [scalar] estimate of d (where d+1 is the order of the fractional integration process)
    # epsFI      : [vector] (1 x t_end) residuals of the fractional integration process fit
    #             The operator (1+L)**{d} is computed by means of its Taylor expansion truncated at order l_(see the note below)
    # coeff      : [vector] (l_+1 x 1) first l_+1 coefficients (including coeff_0=1) are considered in the Taylor expansion
    #Note: If L is the lag operator the residuals are defined as eps = (1-L)**(1+d)X =(1-L)**d dX , where (1-L)**{d} \approx \sum_l=0**{l_} coeff_l L**l

    # options
    # if exist(OCTAVE_VERSION,builtin) == 0
    #     options = optimoptions(lsqnonlin, TolX, 10**-9, TolFun, 10**-9, MaxFunEvals, 1200, MaxIter, 400, Display, off)
    # else:
    #     options = optimset(TolX, 10**-9, TolFun, 10**-9, MaxFunEvals, 1200, MaxIter, 400, Display, off)

    lb = -0.5
    ub = 0.5
    res = least_squares(objective,d0,args=(dx,l_),bounds=(lb,ub),ftol=1e-9,xtol=1e-9)
    d, exitFlag, resNorm = res.x, res.status, None
    epsFI, coeff = FractIntegrProcess(d,dx,l_+1)

    return d, epsFI, coeff, exitFlag, resNorm


def objective(d, dx, l_):
    eps, _ = FractIntegrProcess(d,dx,l_)
    F = npsum(autocorrelation(eps,10)**2)
    return F


def FractIntegrProcess(d,x,l_):
    # estimate a fractional integration process
    #Compute the first l_ coeff and the approximated residuals of a Fractional Integration Process of order d+1

    t_ = x.shape[0]
    l = arange(1,l_)
    coeff = ones((1,l_))
    coeff[0,1:l_] = (-1)**l*gamma(1+d)/gamma(l+1)/gamma(1+d-l)

    eps = zeros((1,t_-l_+1))
    for t in range(l_,t_):
        if t==l_:
            LX = x[t-1::-1]
        else:
            LX = x[t-1:t-l_-1:-1]
        eps[0,t-l_] = coeff@LX.T
    return eps, coeff

