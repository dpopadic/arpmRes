import matplotlib.pyplot as plt

from scipy.misc import comb

plt.style.use('seaborn')


def Central2Raw(mu):
    ## Map central moments into raw moments
    #  INPUTS
    #   mu  : [vector] (length n_ corresponding to order n_) central moments
    #  OPS
    #   mu_ : [vector] (length n_ corresponding to order n_) corresponding raw moments

    n_   = len(mu)
    mu_ = mu

    for n in range(1, n_):
        mu_[n] = ((-1)**(n+1)) * (mu[0]**n)
        for k in range(n-1):
            mu_[n] =  mu_[n] + comb(n,k) * ((-1)**(n-k+1)) * mu_[k] * mu_[0]**(n-k)
        mu_[n] = mu_[n] + mu[n]

    return mu_
