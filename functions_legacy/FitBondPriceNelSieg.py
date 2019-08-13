from collections import namedtuple

import matplotlib.pyplot as plt
from numpy import array, zeros

from scipy.optimize import least_squares, leastsq

from BondPriceNelSieg import BondPriceNelSieg


def FitBondPriceNelSieg(v_bond, tau, c, cn, par_last):
    # Estimating Nelson Siegel model parameters to fit bond prices

    # Initial values: par_last
    p0 = zeros(4)
    p0[0] = par_last.theta1
    p0[1] = par_last.theta2
    p0[2] = par_last.theta3
    p0[3] = par_last.theta4_squared

    # Specify some lower/upper bounds for each parameter:
    lowerBound = array([p0[0] - 2, p0[1] - 2, p0[2] - 2, max(0, p0[3] - 2)])
    upperBound = p0 + array([2, 2, 3, 2])

    # Estimation
    res = least_squares(NSBPFit, p0, args=(v_bond, c, cn, tau), bounds=(lowerBound, upperBound),
                        tr_solver='lsmr', method='trf', loss='soft_l1', jac='3-point',
                        max_nfev=200000, xtol=1e-8, ftol=1e-8, gtol=1e-8)
    p = res.x
    exitFlag = res.status
    residual = res.optimality

    # Output
    par = namedtuple('par', ['theta1', 'theta2', 'theta3', 'theta4_squared', 'exit', 'res'])
    par.theta1 = p[0]
    par.theta2 = p[1]
    par.theta3 = p[2]
    par.theta4_squared = p[3]
    par.exit = exitFlag
    par.res = residual
    return par


# errors
def NSBPFit(p, v_bond, c, cn, tau):
    tmpPar = namedtuple('tmpPar', ['theta1', 'theta2', 'theta3', 'theta4_squared'])
    tmpPar.theta1 = p[0]
    tmpPar.theta2 = p[1]
    tmpPar.theta3 = p[2]
    tmpPar.theta4_squared = p[3]
    v_bond_NS = zeros(len(v_bond))

    for i in range(len(v_bond)):
        v_bond_NS[i] = BondPriceNelSieg(1, c[i], cn[i], tau[i], tmpPar)
    F = v_bond - v_bond_NS
    return F
