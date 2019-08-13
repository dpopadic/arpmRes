from numpy import exp, where


def YieldNelSieg(tau, par):
    # Evaluate Nelson-Siegel yield
    # INPUTS
    # tau: [vector]times to maturity
    # par: [struc] Nelson-Siegel parameters
    #                   par.theta1(level)
    #                   par.theta2 (slope)
    #                   par.theta3(curvature)
    #                   par.theta4_squared(decay)
    # OUTPUT
    # y:   [vector] yields to maturity (for time to maturity = tau) according to the Nelson-Siegel model

    # if tau is a row vector, traspose it (so that the output y is a column vector [n_,1])
    if tau.shape[0] == 1 and tau.shape[1] > 1:
        tau = tau.T

    y = par.theta1 - par.theta2 * ((1 - exp(-par.theta4_squared * tau)) / (par.theta4_squared * tau)) + par.theta3 * (
                (1 - exp(-par.theta4_squared * tau)) / (par.theta4_squared * tau) - exp(-par.theta4_squared * tau))

    y[where(y < 0)] == 10 ** (-15)  # impose lower bound for yields

    return y
