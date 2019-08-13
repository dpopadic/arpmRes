import matplotlib.pyplot as plt
from scipy.optimize import minimize

from FilterStochasticVolatility import FilterStochasticVolatility


def FitStochasticVolatilityModel(y,initpar):
    #maximum likelihood estimation of the SV model parameters
    #
    #For the original R code for the stochastic volatility filter function refer to
    #"R.H. Shumway and D.S. Stoffer, Time Series Analysis and Its Applications:
    #With R Examples", example 6.18.

    opts = {'maxiter': 6*500}

    res=minimize(Linn,initpar,args=(y,), options=opts)
    return res.x,res.fun,res.status,res


def Linn(para, y):
    # innovations likelihood
    phi0=para[0]
    phi1=para[1]
    sQ=para[2]
    alpha=para[3]
    sR0=para[4]
    mu1=para[6-1]
    sR1=para[7-1]
    like,_=FilterStochasticVolatility(y,phi0,phi1,sQ,alpha,sR0,mu1,sR1)
    return like
