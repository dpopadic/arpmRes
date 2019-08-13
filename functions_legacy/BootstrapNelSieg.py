from collections import defaultdict

import matplotlib.pyplot as plt
from numpy import arange, zeros, interp, floor, where

plt.style.use('seaborn')

from YieldNelSieg import YieldNelSieg
from FitBondPriceNelSieg import FitBondPriceNelSieg


def BootstrapNelSieg(Dates, v_bond, b_sched, tau_out, par_start, ttm=None, zero_rates=None):
    # This function fits the Nelson-Siegel model on coupon-bearing bonds
    # and computes yields (and spreads with respect to a reference curve) accordingly
    #
    # INPUTS
    # Dates: vector of dates relative to prices observations
    # v_bond: n_ x t_end matrix containing the dirty prices of coupon bearing bonds
    # b_sched: n_ x 2 vector for each bond, first column: annual coupon second column: expiry date.
    # tau_out: times to maturity (points of the curve for which yields and spreads are returned)
    # par_start: initial guess for NS parameters
    # ttm and zero_rates: Time to maturities and zero rates of the reference curve (needed to compute the spreads)
    #
    # OUTPUTS
    # theta1,theta2,theta3,theta4_squared: Nelson-Siegel parameters
    # tau_end: times to maturity of real bonds
    # y_tau: yields computed at times to maturity given by the input vector tau_out
    # y_real: yields computed at times to maturity of real bonds
    # y_ref_tau: zero rates at times to maturity given by the input vector tau_out
    # y_ref_real: zero rates at times to maturity of real bonds
    # s_tau: spreads computed at times to maturity given by the input vector tau_out
    # s_real: spreads computed at times to maturity of real bonds

    numBonds = v_bond.shape[0]
    t_ = len(Dates)

    # preallocating variables
    e = zeros((numBonds, 1))
    c = zeros(numBonds)
    cn = zeros(numBonds, dtype=int)
    cd1 = zeros((numBonds, 1))
    tau_end = zeros((numBonds, t_))
    mat = defaultdict(dict)

    theta1 = zeros((1, t_))
    theta2 = zeros((1, t_))
    theta3 = zeros((1, t_))
    theta4_squared = zeros((1, t_))

    y_tau = zeros((len(tau_out), t_))
    y_real = zeros((numBonds, t_))
    y_ref_tau = zeros((len(tau_out), t_))
    y_ref_real = zeros((numBonds, t_))
    s_tau = zeros((len(tau_out), t_))
    s_real = zeros((numBonds, t_))

    ## Coupon-bearing bonds

    for t in range(t_):
        day = Dates[t]
        # v_bond: bond

        for i in range(numBonds):
            e[i] = b_sched[i, 1]  # expiry date
            c[i] = 0.5 * b_sched[i, 0]  # semiannual coupon

            cn[i] = int(floor((e[i] - day) / 180)) + 1  # number of missing coupons from day to expiry

            cd1[i] = e[i] - (cn[i] - 1) * 180  # date of the first coupon after day

            tau = (cd1[i] - day) / 360  # tau[0]: time from day to cd1 (years)

            if tau == 0:  # case when day is a coupon payment date
                tau = 0.5
                cn[i] = cn[i] - 1

            tau_end[i, t] = tau + 0.5 * (cn[i] - 1)  # time from day to the last coupon (years)
            # equivalently: tau_end(t,v_bond)=(e((v_bond)-day))/360

            tau = arange(tau, tau_end[i, t] + 0.01, 0.5).T  # times to maturities (coupons)
            mat[i] = tau  # coupons times to maturities reshaped to be used as input of FitBondPriceNelSieg

        ## fitting NS model
        if t == 0:
            par = FitBondPriceNelSieg(v_bond[:, t], mat, c, cn, par_start)
        else:
            par = FitBondPriceNelSieg(v_bond[:, t], mat, c, cn, par)

        par = FitBondPriceNelSieg(v_bond[:, t], mat, c, cn, par_start)

        theta1[0, t] = par.theta1
        theta2[0, t] = par.theta2
        theta3[0, t] = par.theta3
        theta4_squared[0, t] = par.theta4_squared

        ## Yields
        y_tau[:, t] = YieldNelSieg(tau_out, par)
        y_real[:, t] = YieldNelSieg(tau_end[:, t], par)

        if ttm is not None:
            ## Zero rates
            y_ref_tau[:, t] = interp(tau_out, ttm, zero_rates[:, t])
            y_ref_real[:, t] = interp(tau_end[:, t], ttm, zero_rates[:, t])

            ## Spreads
            s_tau[:, t] = y_tau[:, t] - y_ref_tau[:, t]
            s_real[:, t] = y_real[:, t] - y_ref_real[:, t]

    return theta1, theta2, theta3, theta4_squared, tau_end, y_tau, y_real, y_ref_tau, y_ref_real, s_tau, s_real