import numpy as np
from numpy import ones, zeros, prod, where, cov, mean, tile, r_
from numpy import sum as npsum
from numpy.linalg import solve


def IncludeDataMVE(epsi,last=0):
    # This function computes the Minimum Volume Ellipsoid enclosing all data.
    # The location and dispersion parameters that define the ellipsoid are
    # multivariate high-breakdown estimators of location and scatter.
    # INPUTS
    #  epsi        :[matrix](i_ x t_end) dataset of invariants
    #  last        :[scalar] if last!=0 only the last computed mean and covariance are returned
    # OUTPUTS
    #  mu_MVE      :[matrix](i_ x k_) contains the mean vectors computed at each iteration
    #  sigma2_MVE  :[array](i_ x i_ x k_) contains the covariance matrices computed at each iteration
    #  bound       :[row vector] indices of observations near the ellipsoid's bound

    # For details on the exercise, see here .

    ## Code

    t_=epsi.shape[1]

    #step0: initialize
    mu=mean(epsi,1,keepdims=True)
    sigma2=cov(epsi)
    k=0
    mu_MVE=mu
    sigma2_MVE=sigma2[...,np.newaxis]
    det1=0
    w=ones((1,t_))/t_

    KeepLoop=1
    while KeepLoop:
        k=k+1

        #step1: compute z-scores
        Mah2=np.sum((epsi-mu)*solve(sigma2,(epsi-mu)),axis=0)

        #step[1:] update weights
        update=where(Mah2>1)
        w[0,update]=w[0,update]*Mah2[update]

        #step3: update output
        mu=epsi@w.T/npsum(w)
        sigma2=(epsi-tile(mu, (1,t_)))@np.diagflat(w)@(epsi-tile(mu, (1,t_))).T
        mu_MVE = r_['-1',mu_MVE,mu]
        sigma2_MVE=r_['-1',sigma2_MVE,sigma2[...,np.newaxis]]

        #step4: check convergence
        KeepLoop = 1-prod(Mah2<=1)

    bound=where(Mah2>0.98)[0]

    if last!=0:
        mu_MVE=mu_MVE[:,-1]
        sigma2_MVE=sigma2_MVE[:,:,-1]
    return mu_MVE, sigma2_MVE, bound
