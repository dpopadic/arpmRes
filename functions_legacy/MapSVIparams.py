from numpy import array, exp

from scipy.special import erf


def MapSVIparams(p_var_ATM,p_phi):

    # This function converts the theta_1,...,theta_6 parametrization of the SVI model
    # into its original parametrization a,b,c,rho,eta,gamma, found in equation
    # (5.1) of the paper "Arbitrage-free SVI volatility surfaces" by Jim
    # Gatheral and Antoine Jacquiery, version April 6, 2012

    # INPUTS
    #  p_var_ATM [vector]: [2] its three components contain, respectively, theta_1,
    #                          theta_2,theta_3
    #  p_phi [vector]: [2] its three components contain, respectively, theta_4,theta_5,
    #                      theta_6

    # OUTPUTS
    #  par_var_ATM [vector]: [2] its three components contain, respectively, a,
    #                            b,c
    #  par_phi [vector]: [2] its three components contain, respectively, rho,
    #                        eta,gamma

    ## Code
    a = (exp(p_var_ATM[0])+p_var_ATM[1])/4
    b = (exp(p_var_ATM[0])-3*p_var_ATM[1])/4
    c = p_var_ATM[2]-a/5

    rho = erf(p_phi[0])
    eta = (erf(p_phi[1])+1)*1.7/2
    gamma = (erf(p_phi[2])+1)/2

    par_var_ATM = array([a,b, c])
    par_phi = array([rho, eta, gamma])

    return par_var_ATM, par_phi
