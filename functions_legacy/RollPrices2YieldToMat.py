from numpy import log, tile, array, newaxis, squeeze


def RollPrices2YieldToMat(tau, z):
    # Computes yield to maturity given the time to maturity tau and the rolling
    # value z
    # INPUTS
    # tau  :[column vector] vector of times to maturity
    # z    :[matrix] bond rolling prices
    # OUTPUTS
    # y     :[matrix] yields to maturity
    # logz  :[matrix] log rolling prices

    ## Code

    logz=log(z) #compute log rolling prices
    if isinstance(tau, float) or isinstance(tau, int):
        tau = array([tau])
    y=-squeeze(tile(1/tau[...,newaxis], (1,z.shape[1])))*logz #compute yields to maturity
    return y,logz
