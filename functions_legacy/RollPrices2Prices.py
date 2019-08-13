import numpy as np
from numpy import arange, zeros, interp, ceil, floor, abs
from numpy import max as npmax

from intersect_matlab import intersect
from ARPM_utils import datenum


def RollPrices2Prices(t_end_str, tau, dates, z_roll):
    # This function uses rolling values to compute the zero-coupon bond value
    # (i.e., discount factors) with maturities in t_end_str.
    # INPUTS
    #  t_end_str [vector]: (k_ x 1) selected maturities (as .Tdd-mmm-yy.T strings)
    #  tau [vector]: (n_ x 1) times to maturity corresponding to rows of z_roll
    #  Date [vector]: (1 x t_end) dates corresponding to columns of z_roll
    #  z_roll [matrix]: (n_ x t_end) rolling values
    # OUTPUTS
    #  date [cell vector]: (k_ x 1) cell date{j} contains the numerical value of
    #                              dates corresponding to columns of z{j}
    #  z [cell vector]: (k_ x 1) cell z{j} contains the evolution of zero-coupon
    #                           bond value with maturity t_end{j}
    #  t_end [vector]: (k_ x 1) contains the numerical value corresponding to
    #                          date strings in t_end_string

    # tau_int: vector of maturities for interpolation
    tauRange=ceil((dates[-1]-dates[0])/365)
    _, _,tauIndex=intersect(tauRange,tau)
    if not tauIndex:
        tauIndex=tau.shape[0]
    tau_int=arange(tau[0],tau[tauIndex]+tau[0], tau[0])
    # declaration and preallocation of variables
    t_=z_roll.shape[1]
    n_=npmax(tau_int.shape)
    z_roll_int=zeros((n_,t_))
    expiry=zeros((n_,t_))
    expiry_f=zeros((n_,t_))
    k_=t_end_str.shape[0]
    t_end=zeros((k_,1),dtype=int)
    z={}
    date={}

    for t in range(t_):
        # remove zeros
        indexPolished=np.where(abs(z_roll[:,t])>0)[0]
        # matrix of rolling values: z_roll(i,t)=z_{t}(tau[i]+t)
        z_roll_int[:,t]=interp(tau_int,tau[indexPolished],z_roll[indexPolished,t])
        # expiries
        for i in range(n_):
            expiry[i,t]=tau_int[i]*365+dates[t]
            expiry_f[i,t]=floor(expiry[i,t]) # to remove HH:mm:ss

    # zero-coupon bond values (i.e., discount factors) with fixed expiry
    for j in range(k_):
        z[j]=zeros((1,t_))
        date[j]=zeros((1,t_))
        t_end[j]=datenum(t_end_str[j])
        # z[j] = np.where(expiry_f==t_end[j],z_roll_int,z[j])
        # date[j] = np.where(expiry_f==t_end[j],dates,date[j])
        for t in range(t_):
            for i in range(n_):
                if expiry_f[i,t]==t_end[j]:
                   z[j][0,t]=z_roll_int[i,t]
                   date[j][0,t]=dates[t]
        # remove zeros
        indexzeros=np.where(date[j]==0)
        date[j][indexzeros]=np.NAN
        z[j][indexzeros]=np.NaN
    return date, z, t_end
