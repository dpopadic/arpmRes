import matplotlib.pyplot as plt
import numpy as np
from numpy import arange, zeros, log, sqrt

from scipy.stats import t

plt.style.use('seaborn')

from MaxLikelihoodFPLocDispT import MaxLikelihoodFPLocDispT


def CalibDegOfFreedomMLFP(x,FP,maxdf,stepdf):
    # Student t model
    # MLFP for mu and sigma on a grid of degrees of freedom (df) the best fit
    # corresponds to df which gives rise to the highest (log)likelihood L)
    #  INPUTS
    # x       :[vector](1 x t_end) empirical realizations
    # FP      :[vector](1 x t_end) flexible probabilities associated with vector x
    # maxdf   :[scalar] maximum value for nu to be checked
    # stepdf  :[scalar] step between consecutive values of nu to be checked
    #  OPS
    # mu      :[scalar] estimated location parameter
    # sig2    :[scalar] estimated dispersion parameter
    # nu      :[scalar] best degrees of freedom nu

    ## Code
    df=arange(1,maxdf+stepdf,stepdf)
    Tol=10**(-6)
    l_=len(df)

    Mu = zeros((l_,1))
    Sigma2 = zeros((l_,1))
    L = zeros((l_,1))
    for i in range(l_):
        Mu[i],Sigma2[i],_ = MaxLikelihoodFPLocDispT(x,FP,df[i],Tol,1)
        L[i]=FP@log(t.pdf((x - Mu[i]) / sqrt(Sigma2[i]), df[i]) / sqrt(Sigma2[i])).T

    imax= np.argmax(L)

    mu = Mu[imax]
    sig2 = Sigma2[imax]
    nu = df[imax]

    return mu, sig2, nu

