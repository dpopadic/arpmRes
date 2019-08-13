from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import least_squares

from YieldNelSieg import YieldNelSieg


def FitYieldNelSieg(tau,y,par0,lb=None,ub=None):
    # Estimating Nelson Siegel model parameters to fit the yield curve
    # INPUT
    # tau  :[vector] (n_ x 1) times to maturity
    # y    :[vector] (n_ x 1) rates
    # par0 :[vector] initial guess for the vector of parameters (theta1=level,theta2=slope,theta3=curvature, theta4**2=decay)
    # lb   :[vector] (1 x 4) lower bound for each parameter
    # ub   :[vector] (1 x 4) upper bound for each parameter
    # OP
    # par  :[vector] estimated parameters
    #########################################################################

    #make sure the input y is a column vector
    if y.shape[0]==1 and y.shape[1]>1:
        y=y.T

    #lower/upper bounds for each parameter (default: no bounds)

    # Initial values: par_last
    p = [0]*4
    p[0]=par0.theta1	# theta1 (level)
    p[1]=par0.theta2	# theta2 (slope)
    p[2]=par0.theta3	# theta3  (curvature)
    p[3]=par0.theta4_squared	# theta4_squared (decay) =(theta4)**2

    # Optimization options.
    # if exist(OCTAVE_VERSION,builtin)==0
    #     options = optimoptions(lsqnonlin, TolX, 1e-6, TolFun, 1e-06, MaxFunEvals, 600, MaxIter, 400, Display, off,...
    #     DiffMaxChange, 1e-1, DiffMinChange, 1e-8, Algorithm, .Ttrust-region-reflective.T)
    # else:
    #     options = optimset(TolX, 1e-6, TolFun, 1e-06, MaxFunEvals, 600, MaxIter, 400, Display, off)

    # Estimation
    if lb is not None and ub is not None:
        res = least_squares(NSYFit,p,args=(y,tau), bounds=(lb,ub),max_nfev=4*500)
    else:
        res = least_squares(NSYFit, p, args=(y, tau), max_nfev=4 * 500)
    p = res.x

    # Output
    par = namedtuple('par',['theta1','theta2','theta3','theta4_squared', 'exit','res', 'resNorm'])
    par.theta1	= p[0]
    par.theta2	= p[1]
    par.theta3	= p[2]
    par.theta4_squared= p[3]
    par.exit=res.status
    par.res=res.optimality
    # par.resNorm=resNorm
    return par

# errors
def NSYFit(tmpP, y, tau):
    tmpPar = namedtuple('par', ['theta1', 'theta2', 'theta3', 'theta4_squared', 'exit', 'res', 'resNorm'])
    tmpPar.theta1	= tmpP[0]
    tmpPar.theta2	= tmpP[1]
    tmpPar.theta3	= tmpP[2]
    tmpPar.theta4_squared	= tmpP[3]

    y_NS = YieldNelSieg(tau,tmpPar)

    if any(y_NS==np.inf):
        raise ValueError('inf values')
    F= y.flatten() - y_NS.flatten()
    return F
