from collections import namedtuple

from MapSVIparams import MapSVIparams
from numpy import array, zeros
from scipy.optimize import least_squares


def FitVar_ATMSVI(tau,var_mkt_ATM, theta_var_ATM_start):
    # This function calibrates the theta_1,theta_2,theta_3 parameters of the SVI model such
    # that the theoretical variance curve best match the observed at-the-money
    # variance curve. Notice that for the ATM variance, only the first three
    # parameters are needed.

    # INPUTS
    #  tau [vector]: (n_ x 1) times to maturity corresponding to values in
    #                         var_mkt_ATM
    #  var_mkt_ATM [vector]: (n_ x 1) observed at-the-money variance values

    #  theta_var_ATM_start [structure]: starting parameters for fitting
    #                                  Fields: theta_1,theta_2,theta_3

    # OUTPUTS
    #  theta_var_ATM [structure]: SVI parameters theta_1,theta_2,theta_3
    #                            Fields: theta_1,theta_2,theta_3
    #  var_model_ATM [vector]: (n_ x 1) at-the-money variance obtained through
    #                                   SVI model, evaluated at times to
    #                                   maturity in tau

    ## Code
    # initializing var_ATM
    var_ATM = zeros((len(tau),1))

    # Estimation
    par_start = array([theta_var_ATM_start.theta1, theta_var_ATM_start.theta2, theta_var_ATM_start.theta3])
    res = least_squares(objective, par_start, args=(tau, var_mkt_ATM),ftol=1e-10,xtol=1e-10,max_nfev=2*600,verbose=0)
    p = res.x
    exitFlag = res.status
    resNorm = res.optimality
    theta_var_ATM = namedtuple('theta',['theta1', 'theta2', 'theta3'])

    theta_var_ATM.theta1 = p[0]
    theta_var_ATM.theta2 = p[1]
    theta_var_ATM.theta3 = p[2]
    var_ATM = p[0]*tau**2 + p[1]*tau + p[2]
    var_model_ATM = var_ATM/tau
    return theta_var_ATM, var_model_ATM, exitFlag, resNorm


def objective(p, tau, var_mkt_ATM):
    p_tmp,_ = MapSVIparams(p,zeros(3))
    a = p_tmp[0]
    b = p_tmp[1]
    c = p_tmp[2]
    var_ATM = a*tau**2+b*tau+c
    F = var_mkt_ATM.flatten() - var_ATM/tau
    return F
