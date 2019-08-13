import matplotlib.pyplot as plt
from numpy import arange, sign, zeros, log, sqrt, exp
from numpy.linalg import norm
from scipy.stats import norm, t, lognorm

plt.style.use('seaborn')

from ShiftedLNMoments import ShiftedLNMoments


def DiscretizeNormalizeParam(tau, k_, model, par):
    # This function discretizes the one-step normalized pdf when the
    # distribution is parametrically specified
    # INPUTS
    #  tau    :[scalar] projection horizon
    #  k_     :[scalar] coarseness level
    #  model  :[string] specifies the distribution: shiftedLN,.TStudent t.T,Uniform
    #  par    :[struct] model parameters
    # OUTPUTS
    #  xi     :[1 x k_] centers of the bins
    #  f      :[1 x k_] discretized pdf of invariant

    ## Code

    # grid
    a = -norm.ppf(10**(-15),0,sqrt(tau))
    h = 2*a/k_
    xi = arange(-a+h,a+h,h)

    # discretized initial pdf (standardized)
    if model=='shiftedLN':
        m, s,_ = ShiftedLNMoments(par)
        csi = par.c
        mu = par.mu
        sig = sqrt(par.sig2)
        if sign(par.skew)==1:
            M = (m-csi)/s
            f = 1/h*(lognorm.cdf(xi+h/2+M,sig,scale=exp(mu-log(s)))-lognorm.cdf(xi-h/2+M,sig,scale=exp(mu-log(s))))
            f[k_] = 1/h*(lognorm.cdf(-a+h/2+M,sig,scale=exp(mu-log(s)))-lognorm.cdf(-a+M,sig,scale=exp(mu-log(s))) +\
            lognorm.cdf(a+M,sig,scale=exp(mu-log(s)))-lognorm.cdf(a-h/2+M,sig,scale=exp(mu-log(s))))
        elif sign(par.skew)==-1:
            M = (m+csi)/s
            f = 1/h*(lognorm.cdf(-(xi-h/2+M),sig,scale=exp(mu-log(s)))-lognorm.cdf(-(xi+h/2+M),sig,scale=exp(mu-log(s))))
            f[k_-1] = 1/h*(lognorm.cdf(-(-a+M),sig,scale=exp(mu-log(s)))-lognorm.cdf(-(-a+h/2+M),sig,scale=exp(mu-log(s))) +\
            lognorm.cdf(-(a-h/2+M),sig,scale=exp(mu-log(s)))-lognorm.cdf(-(a+M),sig,scale=exp(mu-log(s))))

    elif model=='Student t':
        nu = par
        f = 1/h*(t.cdf(xi+h/2,nu)-t.cdf(xi-h/2,nu))
        f[k_-1] = 1/h*(t.cdf(-a+h/2,nu)-t.cdf(-a,nu) + t.cdf(a,nu)-t.cdf(a-h/2,nu))

    elif model=='Uniform':
        mu = par.mu
        sigma = par.sigma
        f = zeros(k_)
        f[(xi>=-mu/sigma)&(xi<=(1-mu)/sigma)] = sigma
    return xi, f

