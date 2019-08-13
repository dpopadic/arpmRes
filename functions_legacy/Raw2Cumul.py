from scipy.misc import comb


def Raw2Cumul(mu_):
    ## Map raw moments into cumulative moments
    #  INPUTS
    #   mu_ : [vector] (length n_ corresponding to order n_) corresponding raw moments
    #  OPS
    #   ka  : [vector] (length n_ corresponding to order n_) cumulative moments

    n_  = mu_.shape[1]
    ka = mu_.copy()

    for n in range(n_):
        ka[0,n] = mu_[0,n]
        for k in range(n):
            ka[0,n] = ka[0,n] - comb(n,k)*ka[0,k]*mu_[0,n-k-1]

    return ka
