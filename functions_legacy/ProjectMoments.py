from numpy import ones, zeros, sqrt


from Raw2Cumul import Raw2Cumul
from Cumul2Raw import Cumul2Raw
from Raw2Central import Raw2Central


def ProjectMoments(f_1, tau, k_):
    # Project first k_ standardized statistics to horizon tau
    #  INPUTS
    # f_1        :[struct] with fields:
    #             - f_1.x :[vector](1 x t_end) scenarios
    #             - f_1.p :[vector](1 x t_end) flexible probabilities associated with scenarios x
    #                                       default: all equal probabilities
    # tau        :[scalar] projection horizon
    # k_         :[scalar] maximum order of the statistics to compute
    #  OPS
    # gamma_tau  :[vector](1 x k_) standardized statistics up to order k_ to horizon

    # For details on the exercise, see here .

    ## Code

    # compute non-central moments from the distribution f1
    x = f_1.x
    t_ = x.shape[1]
    if f_1.p is None:
        p = ones((1,t_))/t_
    else:
        p = f_1.p
    mu_1 = zeros((1,k_))
    mu_1[0,0] = x@p.T
    if k_>1:
        for k in range(1,k_):
            mu_1[0,k] = (x**(k+1))@p.T

    # compute single-period cumulants
    c_1 = Raw2Cumul(mu_1.copy())

    # compute multi-period cumulants
    c_tau = tau*c_1

    # compute multi-period non-central moments
    mu_tau = Cumul2Raw(c_tau)

    # compute multi-period central moments
    mu_tilde_tau = Raw2Central(mu_tau.copy())

    # compute multi-period standardized statistics
    gamma_tau = mu_tilde_tau.copy()
    if k_>1:
        gamma_tau[0,1] = sqrt(mu_tilde_tau[0,1])
    if k_>2:
        for k in range(2, k_):
            gamma_tau[0,k] = mu_tilde_tau[0,k] / (gamma_tau[0,1]**(k+1))
    if k_>3:
        gamma_tau[0,3] = gamma_tau[0,3]-3# excess kurtosis
    return gamma_tau
