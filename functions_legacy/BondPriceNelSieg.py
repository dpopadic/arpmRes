import matplotlib.pyplot as plt
from numpy import size, zeros, exp, tile, sum as npsum

plt.style.use('seaborn')

from YieldNelSieg import YieldNelSieg


def BondPriceNelSieg(notional,c,cn,tau,par):
    # Evaluate bond price according to Nelson-Siegel model
    # INPUTS
    # notional  [scalar] notional value
    # c:        [vector] amount paid at each coupon payment (for unit of notional). If scalar, the same amount is considered at each payment
    # c_n:      [scalar] number of coupons to be considered
    # tau:      [vector] time interval between the evaluation time and the payment of each coupon
    # par:      [struc] Nelson-Siegel parameters
    #                   par.theta1(level)
    #                   par.theta2 (slope)
    #                   par.theta3(curvature)
    #                   par.theta4_squared(decay)
    # OUTPUT
    # v_bond: coupon bond price according to Nelson-Siegel model
    if size(c)==1:
        c=tile(c,cn)
    c[-1]=1+c[-1]
    v_bond=notional*(c@exp(-tau*YieldNelSieg(tau, par)))
    return v_bond