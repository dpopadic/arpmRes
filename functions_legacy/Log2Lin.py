from numpy import diag, exp


def Log2Lin(mu, sigma2):
    ## Map moments of log-returns to linear returns
    #  INPUTS
    #   mu    : [vector] (n_ x 1)
    #   sigma2 : [matrix] (n_ x n_)
    #  OPS
    #   m     : [vector] (n_ x 1)
    #   s     : [matrix] (n_ x n_)

    m = exp(mu + (1/2)@diag(sigma2)) - 1
    s2 = exp(mu + (1/2)@diag(sigma2))@exp(mu + (1/2)@diag(sigma2)).T * (exp((sigma2)) - 1)
    return m, s2
