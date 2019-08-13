from numpy import ones, abs, mean, log, exp, sqrt, intersect1d, in1d, nonzero, maximum, r_, var, real
from numpy import sum as npsum
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import xlabel, title
from numpy.linalg import solve

from scipy.optimize import fmin
from scipy.special import iv, ive

plt.style.use('seaborn')


def FitCIR_FP(ret, dt, params=None, p=None):
    # Original code from " A stochastic Process Toolkit for Risk Management "
    # by Brigo, Dalessandro, Neugebauer and Triki, 15 November 2007,

    # credit to Brigo, Dalessandro, Neugebauer and Triki

    # ML_CIRparams = [ k s2 eta ]

    t_ = len(ret)

    #p: flexible probabilities if the argument is missing the observations are equally weighted
    if p is None:
        p =(1/t_)*ones((1,t_))

    if params is None:
        x = r_[ones((1,t_-1)), ret[np.newaxis,:t_-1]]
        ols = solve(x@x.T,x@ret[1:t_].T)
        m = mean(ret)
        v = var(ret,ddof=1)
        params = r_[-log(ols[1])/dt, m, sqrt(2*ols[1]*v/m)]

    ML_CIRparams,fevals = fmin(FT_CIR_LL_ExactFull, params, args=(dt,ret,p[0]),maxfun=1000,xtol=1e-8,ftol=1e-8,
                               disp=False, retall=True)
    return ML_CIRparams

def FT_CIR_LL_ExactFull(params,dt,ret,p):
    kappa = params[0]
    s2 = params[1]
    eta = params[2]
    c = (2*kappa)/((eta**2)*(1 - exp(-kappa*dt)))
    q = ((2*kappa*s2 )/(eta**2)) -1
    u = c*exp(-kappa*dt)*ret[:-1]
    v = c*ret[1:]

    mll = npsum(- p[1:]*log(c) + p[1:]*(u+v) - p[1:]*log(v/u)*q/2 -p[1:]*log(ive(q, 2*sqrt(u*v)))
                - p[1:]*abs(real(2*sqrt(u*v))))
    return mll
