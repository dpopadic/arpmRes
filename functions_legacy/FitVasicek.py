from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from numpy import zeros, exp, tile

from scipy.optimize import least_squares


from ZCBondPriceVasicek import ZCBondPriceVasicek


def FitVasicek(tau,y,theta_start):
    # This function estimates Vasicek model parameters to fit market prices
    # INPUTS
    # tau   :[column vector] time to maturity vector
    # y     :[matrix] market yields observations coresponding to different times to maturity
    # theta_start  :[struct] initial values of the parameters
    # OUTPUTS
    # theta_hat  :[struct] contains the estimated parameters, the value of the residuals at the solution and the exit condition

    ## Code

    z = exp(-y*tile(tau, (1,y.shape[1]))) #bond prices

    # Initial values: last prices
    theta = zeros(4)
    theta[0] = theta_start.theta0 # interest rate y0
    theta[1] = theta_start.theta1 # interest rate mean reverting rate k
    theta[2] = theta_start.theta2 # interest rate long term mean mu
    theta[3] = theta_start.theta3 # interest rate volatility sigma

    # Specify some lower/upper bounds for each parameter:
    lb = [-10, 0, -10, 0]
    ub = [np.Inf, np.Inf, np.Inf, np.Inf]

    # Estimation

    res = least_squares(VasicekFit, theta, args=(tau, z), bounds=(lb,ub), max_nfev=4*500)
    theta = res.x

    # Output
    theta_hat = namedtuple('theta',['theta0','theta1','theta2','theta3', 'exit', 'res'])
    theta_hat.theta0	= theta[0]
    theta_hat.theta1	= theta[1]
    theta_hat.theta2	= theta[2]
    theta_hat.theta3	= theta[3]
    theta_hat.exit      = res.status
    theta_hat.res       = res.cost
    return theta_hat


# Objective function
def VasicekFit(tmpP, tau, z):

    tmpPar = namedtuple('theta',['theta0','theta1','theta2','theta3'])
    tmpPar.theta0 = tmpP[0]
    tmpPar.theta1 = tmpP[1]
    tmpPar.theta2 = tmpP[2]
    tmpPar.theta3 = tmpP[3]

    prices = ZCBondPriceVasicek(tau,tmpPar)
    f = z-tile(prices, (1,z.shape[1]))
    return f.flatten()

