import matplotlib.pyplot as plt

from scipy.misc import comb

plt.style.use('seaborn')


def Cumul2Raw(ka):
    ## Map cumulative moments into raw moments
    #  INPUTS
    #   ka  : [vector] (length n_ corresponding to order n_) cumulative moments
    #  OPS
    #   mu_ : [vector] (length n_ corresponding to order n_) corresponding raw moments

    n_   = ka.shape[1]
    mu_ = ka.copy()
    for n in range(n_):
        mu_[0,n] = ka[0,n]
        for k in range(n):
            mu_[0,n] = mu_[0,n] + comb(n, k)*ka[0,k]*mu_[0,n-k-1]
    return mu_
