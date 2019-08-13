import matplotlib.pyplot as plt
import numpy as np
from numpy import min as npmin, max as npmax
from numpy import zeros, cumsum, linspace

plt.style.use('seaborn')

from ARPM_utils import date_mtop


def binningHFseries(time_trades,flag_dt,dx=None):
    # This function computes the number of trades and the traded volume at
    # time intervals of 1-second
    #  INPUT
    # time_trades :[row vector] trade times
    # flag_dt     :[string] to select the frequency of observations
    # dx          :[row vector] traded volume at time_of_trades. It is optional. Input dx only if interested in dv
    #  OP
    # dk  :[row vector] number of trades between t and t+dt
    # k   :[row vector] cumulative number of trades at time t
    # t   :[row vector] time binned with frequency dt
    # dq  :[row vector] traded volume between t and t+dt.

    ## Code

    if dx is None:
        dx = zeros((1,len(time_trades)))

    # vector of time
    time_delta = date_mtop(npmax(time_trades))-date_mtop(npmin(time_trades))

    if flag_dt=='1second':
        T = time_delta.total_seconds() # delta_t = 1 second

    # delta_t = (npmax(time_trades)-npmin(time_trades))/T
    # t = arange(npmin(time_trades),npmax(time_trades), delta_t)
    t = linspace(npmin(time_trades), npmax(time_trades), int(T)+1)

    #compute the number of events for each delta_t
    dk, dq = zeros((1, len(t)-1)), zeros((1, len(t)-1))
    for i in range(len(t)-1):
        if i==len(t)-1:
            dk[0,i] = np.sum((time_trades >= t[i]) & ( time_trades<=t[i+1] ) )
            dq[0,i] = np.sum(dx[0, (time_trades >= t[i])&(time_trades<=t[i+1])])
        else:
            dk[0,i] = np.sum((time_trades >= t[i]) & ( time_trades<t[i+1] ) )
            dq[0,i] = np.sum(dx[0, (time_trades >= t[i])&(time_trades<t[i+1])])
    k = cumsum(dk) #series over delta_t  intervals
    return dk, k, t, dq
