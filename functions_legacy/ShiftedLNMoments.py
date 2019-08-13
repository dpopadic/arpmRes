from numpy import sign, exp, sqrt


def ShiftedLNMoments(par):
    # This function computes the first three central moments from a positive
    # shifted log-normal distribution.
    #  INPUTS
    # par   :[struct] distribution's parameters {sigma2, mu, c} and empirical skewness skew
    #  OPS
    # m     :[scalar] mean
    # s     :[scalar] standard deviation
    # skew  :[scalar] skewness

    ## Code
    csi = par.c
    mu = par.mu
    sig = sqrt(par.sig2)

    m = sign(par.skew)*(csi+exp(mu+sig*sig/2))
    s = exp(mu+sig*sig/2)*sqrt(exp(sig*sig)-1)

    skew = sign(par.skew)*(exp(sig**2)+2)*sqrt(exp(sig**2)-1)

    return m, s, skew
