import matplotlib.pyplot as plt
from numpy import zeros, r_

from scipy.optimize import least_squares

from blsprice import blsprice
from CallPriceHestonFFT import CallPriceHestonFFT

plt.style.use('seaborn')


def FitHeston(tau, k, sigma_impl, r, s_0, z_0):
    # This function estimates Heston model parameters to fit call options.T
    # market prices
    # INPUTS
    # tau         :[column vector] vector of times to maturity
    # k           :[matrix] matrix of strikes
    # sigma_impl  :[matrix] matrix of implied volatilities
    # r           :[scalar] risk free rate
    # s_0         :[scalar] current price of the underlying
    # z_0         :[vector] (1 x 5) initial values of the parameters
    # OUTPUTS
    # z          :[vector] (1 x 5) estimated parameters
    # c_heston   :[matrix] matrix containing the call option prices estimated by means of Heston model
    # exit_flag  :[scalar] value describing the exit condition
    # res_norm   :[scalar] value of the residual

    ## Code
    # new parameter representation to impose the constraint: 2*kappa@sigma_bar**2>lam**2
    # z0_tilde = [ 2*kappa@sigma_bar**2-lam**2, sigma_bar, lam, rho, sigma_0]
    z0_tilde = r_[2 * z_0[0] * z_0[1] ** 2 - z_0[2] ** 2, z_0[1], z_0[2], z_0[3], z_0[4]]

    # constraints (Lower and Upper bounds on Parameters)
    lb = [0.1, 0, 0, -0.99, 0]
    ub = [20, 5, 5, 0.99, 5]

    # optimization
    res = least_squares(objective, z0_tilde, args=(tau, k, sigma_impl, r, s_0), bounds=(lb, ub), max_nfev=100,
                        ftol=1e-6, verbose=0)
    x = res.x
    kappa, sigma_bar, lam, rho, sigma_0 = (x[0] + x[2] ** 2) / (2 * x[1] ** 2), x[1], x[2], x[3], x[4]
    z = r_[kappa, sigma_bar, lam, rho, sigma_0]
    c_heston = zeros((len(tau), k.shape[1]))
    for i in range(len(tau)):
        for j in range(k.shape[1]):
            c_heston[i, j] = CallPriceHestonFFT(s_0, k[i, j], r, tau[i],
                                                r_[kappa, sigma_bar ** 2, lam, rho, sigma_0 ** 2])
    return z, c_heston


def objective(x, tau, k, sigma_impl, r, s_0):
    sigma_bar = x[1]
    lam = x[2]
    rho = x[3]
    sigma_0 = x[4]
    kappa = (x[0] + lam ** 2) / (2 * sigma_bar ** 2)
    f_tmp = zeros((len(tau), k.shape[1]))
    c_heston = zeros((len(tau), k.shape[1]))
    for j in range(len(tau)):
        c_heston[j, :k.shape[1]] = CallPriceHestonFFT(s_0, k[j, :], r, tau[j],
                                                      r_[kappa, sigma_bar ** 2, lam, rho, sigma_0 ** 2])
        f_tmp[j, :k.shape[1]] = blsprice(s_0, k[j, :], r, tau[j], sigma_impl[j, :]) - c_heston[j, :k.shape[1]]
    f = f_tmp.flatten()
    return f
