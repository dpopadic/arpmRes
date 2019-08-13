from numpy import zeros, cumsum, sqrt
from numpy.random import randn

from IG import IG


def NIG(th,k,s,ts,j_):
    # This functions computes j_ simulated paths following a
    # normal-inverse-Gaussian process.
    #  INPUTS
    # th  :[scalar] first parameter
    # k   :[scalar] second parameter
    # s   :[scalar] third parameter
    # ts  :[vector] time vector, with ts[0]=0
    # j_  :[scalar] number of simulations
    #  OPS
    # x   :[matrix](j_ x len(ts)) matrix of simulated paths

    ## Code

    t_=len(ts)

    dx=zeros((j_,t_))
    for t in range(1,t_):
        dt=ts[t]-ts[t-1]
        l=1/k*(dt**2)
        m=dt
        ds=IG(l,m,j_)
        n=randn(j_,1)

        dx[:,[t]]=s*n*sqrt(ds)+th*ds
    x=cumsum(dx,1)
    return x
