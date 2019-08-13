import matplotlib.pyplot as plt
from numpy import zeros
from scipy.stats.stats import pearsonr

plt.style.use('seaborn')


def autocorrelation(x,lag_):
    #x: (1 x t_end) time series from a weakly stationary process
    #lag_: (scalar) number of lags
    #rho: (1 x (lag_+1)) autocorrelations from 0 to lag_ number of lags
    rho=zeros((1,lag_+1))
    rho[0,0],_ = pearsonr(x[0],x[0])
    for k in range(1,lag_+1):
        a=x[0,:-k].T
        b=x[0,k:].T
        rho[0,k], _ = pearsonr(a, b)
    return rho