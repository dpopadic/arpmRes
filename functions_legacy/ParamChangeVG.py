from numpy import sqrt


def ParamChangeVG(theta,sig,nu):
    # This function computes the {c,m,g} parameterization starting from the
    # {theta,sig,nu} parameterization of a Variance-Gamma process.
    # In the {c,m,g} parametrization the log(cf) reads:
    # log(cf[w]) = c@[log(m/(m-iw)) + log(g/(g+iw))]
    #  INPUTS
    # theta  :[scalar]
    # sig    :[scalar]
    # nu     :[scalar]
    #  OPS
    # c      :[scalar]
    # m      :[scalar]
    # g      :[scalar]

    ## Code

    c = 1/nu
    m = -theta/sig**2 + sqrt(theta**2/sig**4 + 2/(sig**2*nu))
    g = theta/sig**2 + sqrt(theta**2/sig**4 + 2/(sig**2*nu))

    return c, m, g
