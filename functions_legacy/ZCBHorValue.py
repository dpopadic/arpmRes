from numpy import exp
from scipy.interpolate import interp1d

from PerpetualAmericanCall import PerpetualAmericanCall


def ZCBHorValue(X, KeyRates, t, u, method=None, varargin=None):
    #This function compute the Zero-coupon bond value at the horizon
    # INPUTs
    # X         :[vector]  (n_ x j_) panel of risk drivers (yields to maturity or shadow rates)
    # KeyRates  :[vector]  (n_ x 1) maturities
    # t         :[scalar]  zero-coupon bond evaluation time
    # u         :[scalar]  zero-coupon bond maturity
    # method    :[string]  if method=[] yields to maturity are the risk drivers
    #                      if method='shadow rates' shadow rates are the risk drivers
    # eta       :[scalars] inverse-call transformation parameter
    # OP
    # Z         :[vector] (1 x j_) panel of zero-coupon bond prices

    xi = u-t
    interp = interp1d(KeyRates.flatten(),X,axis=0,fill_value='extrapolate')
    Y = interp(xi)

    if method=='shadow rates':
        eta = varargin
        ctY = PerpetualAmericanCall(Y,eta)
        Z = exp(-xi*ctY)
    else:
        Z = exp(-xi*Y)

    return Z
