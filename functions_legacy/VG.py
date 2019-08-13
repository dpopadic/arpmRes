from collections import namedtuple

from numpy import zeros, cumsum, sqrt
from numpy.linalg import norm
from scipy.stats import norm

from StochTime import StochTime


def VG(theta,sigma,nu,ts,j_):
    # This function computes j_ simulated paths from a Variance-Gamma process.
    #  INPUTS
    # theta  :[scalar]
    # sigma  :[scalar]
    # nu     :[scalar]
    # ts     :[vector] time vector, with ts[0]=0
    # j_     :[scalar] total number of simulated paths
    #  OPS
    # X      :[matrix](j_ x len(ts)) simulated paths
    # T      :[matrix](j_ x len(ts)) simulated stochastic times

    ## Code
    k_= len(ts)
    input_process = namedtuple('params',['nu', 'j_'])
    input_process.nu = nu
    input_process.J = j_

    dX = zeros((j_,k_))
    dT = zeros((j_,k_))
    for k in range(1,k_):
        dt = ts[k]-ts[k-1]
        dT[:,[k]] = StochTime(dt, 'VG', input_process)
        dX[:,k] = norm.rvs(theta*dT[:,k],sigma*sqrt(dT[:,k]))

    X=cumsum(dX,1)
    T=cumsum(dT, 1)
    return X, T
