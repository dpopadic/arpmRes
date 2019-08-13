from numpy import sqrt


def ShiftedVGMoments(c, theta, sigma, nu, tau):
    # This function computes the first 4 moments for the shifted VG distribution
    #  INPUTS
    # c       :[scalar] drift parameter
    # theta   :[scalar]
    # sigma   :[scalar]
    # nu      :[scalar]
    # tau       :[row vector] vector of horizons
    #  OPS
    # mu      :[scalar] mean
    # sigma2  :[scalar] variance
    # skew    :[scalar] skewness
    # kurt    :[scalar] kurtosis

    ## Code

    mu = c * tau + theta * tau
    sigma2 = (theta ** 2 * nu + sigma ** 2) * tau
    thirdCentralM = (2 * theta ** 3 * nu ** 2 + 3 * sigma ** 2 * theta * nu) * tau
    forthCentralM = (3 * sigma ** 4 * nu + 12 * sigma ** 2 * theta ** 2 * nu ** 2 + 6 * theta ** 4 * nu ** 3) * tau + (
                3 * sigma ** 4 + 6 * sigma ** 2 * theta ** 2 * nu + 3 * theta ** 4 * nu ** 2) * tau ** 2

    skew = thirdCentralM / sqrt(sigma2) ** 3
    kurt = forthCentralM / sigma2 ** 2
    return mu, sigma2, skew, kurt
