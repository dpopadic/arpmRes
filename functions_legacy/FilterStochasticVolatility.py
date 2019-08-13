import matplotlib.pyplot as plt
from numpy import zeros, log, exp, sqrt

plt.style.use('seaborn')


def FilterStochasticVolatility(y, phi0, phi1, sQ, alpha, sR0, mu1, sR1):
    # INPUTS
    # y: [vector] (1 x t_end) is the time series of log(return**2)
    # phi0,phi1,sQ,alpha,sR0,mu1,sR1: [scalars]: parameters of the stochastic volatility model
    # OUTPUTS
    # likelihood: [scalar] -log(likelihood)
    # xp: [vector] (1 x t_end) log of the squared-hidden volatility

    # For the original R code for the stochastic volatility filter function refer to
    # "R.H. Shumway and D.S. Stoffer, Time Series Analysis and Its Applications:
    # With R Examples", example 6.18.
    ###########################################################################

    # Initialize
    t_ = len(y)
    Q = sQ ** 2
    R0 = sR0 ** 2
    R1 = sR1 ** 2
    xf = 0  # =h_0**0
    Pf = sQ ** 2 / (1 - phi1)  # =P_0**0
    if Pf < 0:
        Pf = 0  # make sure Pf not negative
    xp = zeros((1, t_))  # =h_t**t-1
    Pp = zeros((1, t_))  # =P_t**t-1
    pi1 = .5  # initial mix probs
    pi0 = .5
    fpi1 = .5
    fpi0 = .5
    likelihood = 0  # -log(likelihood)
    #

    xp = zeros(t_)
    Pp = zeros(t_)
    for i in range(t_):
        xp[i] = phi1 * xf + phi0
        Pp[i] = phi1 * Pf * phi1 + Q
        sig1 = Pp[i] + R1  # innov var
        sig0 = Pp[i] + R0
        k1 = Pp[i] / sig1
        k0 = Pp[i] / sig0
        e1 = y[i] - xp[i] - mu1 - alpha
        e0 = y[i] - xp[i] - alpha
        #
        den1 = (1 / sqrt(sig1)) * exp(-.5 * e1 ** 2 / sig1)
        den0 = (1 / sqrt(sig0)) * exp(-.5 * e0 ** 2 / sig0)
        denom = pi1 * den1 + pi0 * den0
        fpi1 = pi1 * den1 / denom
        fpi0 = pi0 * den0 / denom
        #
        xf = xp[i] + fpi1 * k1 * e1 + fpi0 * k0 * e0
        Pf = fpi1 * (1 - k1) * Pp[i] + fpi0 * (1 - k0) * Pp[i]
        likelihood = likelihood - 0.5 * log(pi1 * den1 + pi0 * den0)
    return likelihood, xp
