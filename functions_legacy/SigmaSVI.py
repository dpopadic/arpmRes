import numpy as np
from numpy import sqrt, tile

from MapSVIparams import MapSVIparams


def SigmaSVI(tau,m,y,theta_var_ATM,theta_phi):

    # This function computes the value of the volatility according to the SVI
    # model given its theta_1,...,theta_6 parametrization. It is based on the SVI
    # volatility surface presented in equation (5.1) of the paper
    # "Arbitrage-free SVI volatility surfaces" by Jim Gatheral and Antoine
    # Jacquiery, version April 6, 2012

    # INPUTS
    #  tau [vector]: (n_ x 1) times to maturity
    #  m [matrix]: (n_ x k_) m(i,j) correspond to the value of m-moneyness for
    #                               time to maturity tau[i] and an implicit
    #                               value of strike k[j]
    # y [vector]: (n_ x 1) risk-free rates corresponding to times to maturity
    #                      in tau
    #  theta_var_ATM [structure]: SVI parameters theta_1,theta_2,theta_3
    #                            Fields: theta_1,theta_2,theta_3
    #  theta_phi [structure]: SVI parameters theta_4,theta_5,theta_6
    #                        Fields: theta_4,theta_5,theta_6

    # OUTPUTS
    #  sigma_model [matrix]: (n_ x k_) volatility obtained from the SVI model,
    #                                  corresponding to vaues in m

    ## Code
    tau=tile(tau[...,np.newaxis], (1,m.shape[1]))

    # From theta_1 theta_2 ...theta_6 parameterization to the original paper's parameterization

    arg1 = [theta_var_ATM.theta1,theta_var_ATM.theta2,theta_var_ATM.theta3]
    arg2 = [theta_phi.theta4,theta_phi.theta5,theta_phi.theta6]
    par_var_ATM, par_phi = MapSVIparams(arg1, arg2)

    # par_var_ATM
    a = par_var_ATM[0]
    b = par_var_ATM[1]
    c = par_var_ATM[2]
    lam = a*tau**2+b*tau+c # i.e., the at-the-money forward variance var_ATM

    # par_phi
    rho = par_phi[0]
    eta = par_phi[1]
    gamma = par_phi[2]
    phi = eta/((lam**(gamma))*((1+lam)**(1-gamma)))

    # SVI formula for sigma
    # from m-moneyness to k
    k = -sqrt(tau)*m-y*tau

    w = lam/2*(1+rho*phi*k+sqrt((phi*k+rho)**2+(1-rho**2)))
    sigma_model = sqrt(w/tau)
    return sigma_model
