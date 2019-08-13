import matplotlib.pyplot as plt
from numpy import min as npmin
from numpy import where, cov, diag, mean, sqrt, tile
from numpy.linalg import solve

plt.style.use('seaborn')


def FarthestOutlier(epsi,p):
    # This function finds the farthest outlier within the time series epsi,
    # considering the vector of Flexible Probabilities p.
    # INPUTS
    #  epsi     :[matrix](i_ x t_end) time series of invariants
    #  p        :[vector](1 x t_end) vector of Flexible Probabilities
    # OUTPUTS
    #  t_tilde  :[scalar] index of the farthest outlier

    # For details on the exercise, see here .

    ## Code
    t_=epsi.shape[1]
    t_index=range(t_)

    #step1: compute HFP-mean/cov
    mu=mean(epsi,1,keepdims=True)
    sigma2=cov(epsi,ddof=1)

    #step[1:] normalize observations
    z=sqrt(1/t_)*(epsi-tile(mu, (1,t_))).T

    #step3: compute FP-info matrix
    h=z@solve(sigma2,z.T)

    #step4: determine singularities
    h=diag(h).T

    #do not consider singularities
    h = where(h==1,0,h)
    t_index = where(h==1,0,t_index)
    p = where(h==1,0,p)

    #step5: determine outlier
    a=(1-h)/(1-p)
    t_tilde=t_index[a[0]==npmin(a)]
    return t_tilde