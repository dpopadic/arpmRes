import numpy as np
from numpy import arange, array, ones, zeros, empty, sort, where, argsort, cumsum, interp, percentile, squeeze, ceil, floor, diff, linspace, cov, diag, diagonal, eye, abs, round, mean, log, exp, sqrt, tile, r_, nan, intersect1d, in1d, nonzero, maximum
from numpy import sum as npsum, min as npmin, max as npmax
from numpy.linalg import eig, solve, norm
from numpy.random import rand, randn
from numpy.random import multivariate_normal as mvnrnd

from scipy.optimize import minimize
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, bar, scatter, legend, show, savefig, xlim, ylim, scatter, subplots, ylabel, xlabel, title
plt.style.use('seaborn')

from PerpetualAmericanCall import PerpetualAmericanCall


def BondPrice(X, tau_X, c_k, tau_k, principal, method=None, varargin={}):
    # Exact pricing function for bonds
    #  INPUTS
    #   X         [matrix]: (n_ x j_) panel of risk drivers (yields to maturity or shadow rates)
    #   tau_X     [vector]: (n_ x 1) time to maturity associated with the risk drivers
    #   c_k       [vector]: (1 x k_) coupons
    #   tau_k     [vector]: (1 x k_) time to coupon payments
    #   principal [scalar]: principal
    #   method    [string]: if method=[] yields to maturity are the risk drivers
    #                       if method='shadow rates' shadow rates are the risk
    #                       drivers
    # varargin is [
    #   eta       [scalar]: inverse-call transformation parameter.
    #  OPS
    #   V_bond         [matrix]: (1_ x j_) panel of bond prices

    # interpolation
    interp = interp1d(tau_X.flatten(), X, axis=0,fill_value='extrapolate')
    Y = np.atleast_2d(interp(tau_k))
    if isinstance(Y,float):
        Y = array([[Y]])

    ctY = zeros((len(tau_k),Y.shape[1]))
    Z = zeros((len(tau_k),Y.shape[1]))

    for k in range(len(tau_k)):
        if method =='shadow rates':
            ctY[k, :] = PerpetualAmericanCall(Y[k,:],varargin)
            Z[k, :] = exp(-tau_k[k]*ctY[k,:])
        else:
            Z[k, :] = exp(-(tau_k[k]*Y[k,:]))

    V_bond = c_k@Z + principal*Z[-1,:]
    return V_bond
