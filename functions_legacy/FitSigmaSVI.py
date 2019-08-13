from collections import namedtuple

import matplotlib.pyplot as plt
from numpy import zeros, sqrt

from scipy.interpolate import interp1d

plt.style.use('seaborn')

from Delta2MoneynessImplVol import Delta2MoneynessImplVol
from FitVar_ATMSVI import FitVar_ATMSVI
from FitSigmaSVIphi import FitSigmaSVIphi


def FitSigmaSVI(tau, delta, sigma_delta, y, theta_var_ATM_start, theta_phi_start):
    # This function calibrates the theta_1,...,theta_6 parameters of the SVI model such
    # that the theoretical volatility surface curve best match the observed
    # volatility surface.

    # INPUTS
    #  tau [vector]: (n_ x 1) times to maturity corresponding to rows of
    #                         sigma_delta
    #  delta [vector]: (1 x k_) delta-moneyness corresponding to columns of
    #                           sigma_delta
    #  sigma_delta [matrix]: (n_ x k_) observed volatility surface
    #  y [vector]: (n_ x 1) risk-free rates corresponding to times to maturity
    #                       in tau
    #  theta_var_ATM_start [structure]: starting parameters for fitting
    #                                  Fields: theta_1,theta_2,theta_3
    #  theta_phi_start [structure]: starting parameters for fitting
    #                              Fields: theta_4,theta_5,theta_6
    # OUTPUTS
    #  theta_var_ATM [structure]: SVI parameters theta_1,theta_2,theta_3
    #                            Fields: theta_1,theta_2,theta_3
    #  theta_phi [structure]: SVI parameters theta_4,theta_5,theta_6
    #                        Fields: theta_4,theta_5,theta_6

    ## Code
    n_ = len(tau)
    # compute the ATM forward m-moneyness
    if isinstance(y,float) or isinstance(y,int):
        m_ATM = -y*sqrt(tau)
    else:
        m_ATM = -y@sqrt(tau)

    sigma_m_ATM = zeros((n_,1))

    for i in range(n_):
        sigma_m,m = Delta2MoneynessImplVol(sigma_delta[i,:],delta,tau[i],y)
        # compute the  ATM forward variance

        # pchip = PchipInterpolator(m,sigma_m)
        pchip = interp1d(m,sigma_m)
        sigma_m_ATM[i,0] = pchip(m_ATM[i])

    # fit the ATM forward variance
    var_mkt_ATM = sigma_m_ATM**2
    theta_var_ATM, _, ATMexitFlag, ATMresNorm = FitVar_ATMSVI(tau,var_mkt_ATM,theta_var_ATM_start)

    # fit the variance surface
    theta_phi, _, exitFlag, resNorm = FitSigmaSVIphi(tau,delta,sigma_delta,y,theta_var_ATM,theta_phi_start)
    output_fit = namedtuple('output','ATMexitFlag ATMresNorm exitFlag_phi resNorm')
    output_fit.ATMexitFlag = ATMexitFlag
    output_fit.ATMresNorm = ATMresNorm
    output_fit.exitFlag_phi = exitFlag
    output_fit.resNorm = resNorm
    return theta_var_ATM, theta_phi, output_fit

