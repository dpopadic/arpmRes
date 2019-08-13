from numpy import sum as npsum
from numpy import zeros, r_, newaxis, arange, delete
from numpy.linalg import det

from IncludeDataMVE import IncludeDataMVE
from FarthestOutlier import FarthestOutlier


def HighBreakdownFP(epsi,p,last=0,c=0.5):
    # This function computes the High Breakdown Point with Flexible
    # Probabilities estimators of location and dispersion, using an iterative
    # algorithm.
    # INPUTS
    #  epsi         :[matrix](i_ x t_end) invariants
    #  p            :[vector](1 x t_end) Flexible Probabilities associated with the invariants
    #  last         :[scalar] if last!=0 only the last computed mean and covariance are returned
    #  c            :[scalar] enclosed probability: the ellipsoid includes a
    #  probability of at least c the breakdown point is approx. 1-c (when the effective number of scenarios is large)
    # OUTPUTS
    #  mu_HBFP      :[matrix](i_ x k_) contains the mean vectors computed at each iteration
    #  sigma2_HBFP  :[array](i_ x i_ x k_) contains the covariance matrices computed at each iteration
    #  p_tilde      :[vector](1 x k_) vector containing the probability enclosed by the ellipsoid at each iteration
    #  v            :[vector](1 x k_) vector containing the volumes of the ellipsoids computed at each iteraion
    #  t_out        :[vector](1 x k_)vector containing the index of the outliers detected at each iteration

    # For details on the exercise, see here .

    ## Code

    i_,t_ = epsi.shape
    p_tilde = zeros((1,1))
    mu_HBFP=zeros((i_,1))
    sigma2_HBFP=zeros((i_,i_,1))
    v=zeros(1)
    t_out=zeros(1)

    #step0: initialize
    k=0
    T=arange(t_)

    while k==0 or p_tilde[k]>=c:
        k=k+1

        #step1: probability
        p_tilde=r_[p_tilde, npsum(p,keepdims=True)]

        #step2: fit min-vol ellipsoid

        new_mu_HBFP, new_sigma2_HBFP, *_ = IncludeDataMVE(epsi, 1)
        mu_HBFP = r_['-1',mu_HBFP,new_mu_HBFP[...,newaxis]]
        sigma2_HBFP = r_['-1',sigma2_HBFP,new_sigma2_HBFP[...,newaxis]]

        #step3: volume
        v=r_[v, det(sigma2_HBFP[:,:,k])]

        #step4: compute outlier
        t_tilde=FarthestOutlier(epsi,p/p_tilde[k])

        #step5: remove outlier
        t_out = r_[t_out,T[t_tilde]]
        T = delete(T,t_tilde)
        epsi = delete(epsi,t_tilde,axis=1)
        p = delete(p,t_tilde,axis=1)
        print('\r{:.2f} %'.format(min(100*(1-p_tilde[-1]).squeeze()/(1-c),100)), end='',flush=True)

    # Step 6. As soon as p_tilde<c, return the output for index k-1 (so that
    # the enclosed probability corresponding to the output is p_tilde>=c)
    mu_HBFP = mu_HBFP[:,:-1]
    sigma2_HBFP = sigma2_HBFP[:,:,:-1]
    p_tilde = p_tilde[:-1]
    v = v[:-1]
    t_out = t_out[:-2] #the outliers computed at the steps k and (k-1) of the loop are included in the ellipsoid defined by the HBFP parameters at the (k-1)th step, so they have to be removed from the set of the outliers

    if last!=0:
        return mu_HBFP[:,-1], sigma2_HBFP[:,:,-1], p_tilde[-1], v[-1], t_out[-1]
    else:
        return mu_HBFP, sigma2_HBFP, p_tilde, v, t_out
