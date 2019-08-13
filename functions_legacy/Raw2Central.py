import matplotlib.pyplot as plt

from scipy.misc import comb

plt.style.use('seaborn')


def Raw2Central(mu_):
    ## Map raw moments into central moments
    #  INPUTS
    #   mu_ : [vector] (length n_ corresponding to order n_) corresponding raw moments
    #  OPS
    #   mu  : [vector] (length n_ corresponding to order n_) central moments

    n_  = mu_.shape[1]
    mu = mu_.copy()

    for n in range(2,n_+1):
        mu[0,n-1] = (-1.)**n*mu_[0,0]**n
        for k in range(1,n):
            mu[0,n-1] = mu[0,n-1] + comb(n,k)*((-1)**(n-k))*mu_[0,k-1]*(mu_[0,0])**(n-k)
        mu[0,n-1] = mu[0,n-1] + mu_[0,n-1]

    return mu
