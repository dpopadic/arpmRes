from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from numpy import sqrt, tile
from numpy.linalg import norm
from scipy.optimize import least_squares
from scipy.stats import norm

plt.style.use('seaborn')

from SigmaSVI import SigmaSVI


def FitSigmaSVIphi(tau, delta, sigma, y, theta_var_ATM, theta_phi_start):
    # Fit the stochastic volatility inspired model
    # This function calibrates the theta_4,theta_5,theta_6 parameters of the SVI model such
    # that the theoretical volatility surface best match the observed
    # volatility surface. Notice that the theta_1,theta_2,theta_3 parameters of the SVI
    # model are a required input.

    # INPUTS
    #  tau [vector]: (n_ x 1) times to maturity corresponding to rows of sigma
    #  delta [vector]: (1 x k_) delta-moneyness corresponding to columns of
    #                           sigma
    #  sigma [matrix]: (n_ x k_) observed volatility surface
    #  y [vector]: (n_ x 1) risk-free rates corresponding to times to maturity
    #                       in tau
    #  theta_var_ATM [structure]: SVI parameters theta_1,theta_2,theta_3
    #                            Fields: theta_1,theta_2,theta_3
    #  theta_phi_start [structure]: starting parameters for fitting
    #                              Fields: theta_4,theta_5,theta_6

    # OUTPUTS
    #  theta_phi [structure]: SVI parameters theta_4,theta_5,theta_6
    #                        Fields: theta_4,theta_5,theta_6
    #  sigma_model [matrix]: (n_ x k_) volatility obtained from the SVI model,
    #                                  sigma_model(i,j) is the volatility at
    #                                  tau[i] and delta([j])

    ## Code
    n_ = len(tau)
    k_ = len(delta)

    # from delta-moneyness to m-moneyness
    m = tile(norm.ppf(delta), (n_, 1)) * sigma - (y + sigma ** 2 / 2) * tile(sqrt(tau[..., np.newaxis]), (1, k_))

    # Estimation
    par_start = [theta_phi_start.theta4, theta_phi_start.theta5, theta_phi_start.theta6]

    res = least_squares(objective, par_start, args=(tau, m, y, theta_var_ATM, sigma), ftol=1e-9, xtol=1e-9,
                        max_nfev=2 * 600)
    p = res.x
    exitFlag = res.status
    resNorm = res.optimality
    theta_phi = namedtuple('theta', ['theta4', 'theta5', 'theta6'])
    theta_phi.theta4 = p[0]
    theta_phi.theta5 = p[1]
    theta_phi.theta6 = p[2]
    sigma_model = SigmaSVI(tau, m, y, theta_var_ATM, theta_phi)
    return theta_phi, sigma_model, exitFlag, resNorm


def objective(p, tau, m, y, theta_var_ATM, sigma):
    par = namedtuple('theta', ['theta4', 'theta5', 'theta6'])
    par.theta4 = p[0]
    par.theta5 = p[1]
    par.theta6 = p[2]
    sigma_model = SigmaSVI(tau, m, y, theta_var_ATM, par)

    F = sigma_model - sigma
    F = F.flatten()
    return F

