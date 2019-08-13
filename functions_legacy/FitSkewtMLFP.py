import matplotlib.pyplot as plt
import numpy as np
from numpy import log, sqrt
from numpy import sum as npsum
from scipy.optimize import minimize
from scipy.stats import t

plt.style.use('seaborn')

from FPmeancov import FPmeancov


def FitSkewtMLFP(x, p):
    # This function estimates parameters [mu, sigma, alpha, nu] by minimizing
    # the relative entropy between the HFP-pdf specified by the inputs and the
    # univariate pdf of a Skew t-distribution.
    #  INPUTS
    #   x      :[vector](1 x t_end) time series of observations
    #   p      :[vector](1 x t_end) Flexible Probabilities profile
    #  OPS
    #   mu     :[scalar] location parameter
    #   sigma  :[scalar] dispersion parameter
    #   alpha  :[scalar] skew parameter
    #   nu     :[scalar] degrees of freedom

    # initial guess
    m,s2=FPmeancov(x,p)
    parmhat = [m[0,0], sqrt(s2[0,0]), 0, 30]

    lb = [-np.inf, 0, -np.inf, 0]
    ub = [None, None, None, None]
    bound = list(zip(lb,ub))
    options = {'disp': True}
    # Minimize relative entropy
    res = minimize(RelativeEntropySkewt,parmhat,args=(x,p),bounds=bound, method='SLSQP', tol=10**-8,options=options)
    parmhat = res.x

    mu = parmhat[0]
    sigma = parmhat[1]
    alpha = parmhat[2]
    nu = parmhat[3]
    return mu, sigma, alpha, nu


def RelativeEntropySkewt(par,x,p):
    # par -> vector [mu, sigma, alpha, nu]
    x_bar = (x-par[0])/par[1]
    x_tilde = par[2] * x_bar * sqrt((par[3]+1)/(par[3]+x_bar**2))

    skewt = (2/par[1]) * t.pdf(x_bar,par[3]) * t.cdf(x_tilde,(par[3]+1))

    lnSkewt = log(skewt)

    re = npsum(-lnSkewt*p)
    return re
