import matplotlib.pyplot as plt
import numpy as np
from numpy import arange, ones, zeros, where, exp
from numpy import sum as npsum, max as npmax

from scipy.linalg import expm

plt.style.use('seaborn')


def FitMarkovChainFP(dates,t_end,N,n,m,lam=None):
    # This function computes the credit transition matrix p, together with its
    # generator q, starting from the aggregate risk drivers N, n, m.
    #  INPUTS
    # dates   :[vector] vector of dates corresponding to credit migrations (expressed in "num" format)
    # t_end   :[string] time windows's end date (expressed in "date" format)
    # N       :[cell] N{t}[i] is the number of obligors with rating i at time dates[t]
    # n       :[cell] n{t}(i,j) is the cumulative number of transitions between ratings i and j up to time dates[t]
    # m       :[cell] m{t}(i,j) is the number of transitions occured at time dates[t] between ratings i and j
    # lam  :[scalar] half-life exponential decay coefficient (half life expressed in years)
    #  OUTPUTS
    # p       :[matrix] transition matrix (yearly probabilities)
    # q       :[matrix] transition matrix generator
    # num     :[matrix] numerators of matrix q
    # den     :[matrix] denominators of matrix q

    ## Code

    t_ = len(dates)
    r_ = N[-1].shape[0]
    delta_t = zeros(t_)

    # estimation
    num = zeros((r_,r_))
    den = zeros((r_,r_))
    q = zeros((r_,r_))
    if lam is None: # MLE
        for t in range(t_-1):
            delta_t[t] = (np.busday_count(dates[t],dates[t+1]))/252# measured in years
        delta_t[t_] = (np.busday_count(dates[t_],t_end)-1)/252
    else: # MLE with FP
        for t in range(t_):
            delta_t[t] = -(np.busday_count(dates[t],t_end))/252
        delta_t = exp(lam*delta_t)

    for i in range(r_):
        for j in range(r_):
            if i!=j:
                if lam is None: # MLE
                    # numerator
                    num[i,j] = n[-1,i,j]
                    # denominator
                    for t in range(t_):
                        den[i,j] = den[i,j] + N[t,i]*delta_t[t]
                else: # MLE with FP
                    # numerator and denominator
                    for t in range(1,t_):
                        num[i,j] = num[i,j] + m[t,i,j]*delta_t[t]
                        den[i,j] = den[i,j] + N[t-1,i]*(delta_t[t]-delta_t[t-1])
                    num[i,j] = lam*num[i,j]
                    den[i,j] = den[i,j] + N[t_-1,i]*(1-delta_t[t_-1])
                q[i,j] = num[i,j]/den[i,j]

    q[-1,:] = 0

    for i in range(r_):
        q[i,i] = -npsum(q[i,:])

    p = expm(q)
    return p, q, num, den


def workdaysdiff(Date1, Date2):
    #adapted from MATLAB function wrkdydif to guarantee OCTAVE compatibility

    NumberHolidays = zeros((Date1.shape))

    # Get the size of all input arguments scale up any scalars
    sz = [Date1.shape, Date2.shape, NumberHolidays.shape]

    if len(Date1) == 1:
        Date1 = Date1@ones((npmax(sz[:, 0]), npmax(sz[:, 1])))

    if len(Date2) == 1:
        Date2 = Date2@ones((npmax(sz[:, 0]), npmax(sz[:, 1])))

    if len(NumberHolidays) == 1:
        NumberHolidays = NumberHolidays@ones((npmax(sz[:, 0])), npmax(sz[:, 1]))

    # Get the shape of the inputs to reshape output later
    Date1 = Date1.flatten()
    Date2 = Date2.flatten()
    NumberHolidays = NumberHolidays.flatten()
    NumNumberDays = len(Date1)
    NumberDays = zeros((NumNumberDays, 1))

    # Output

    Step = ones((Date1.shape))
    Step = where(Date1 > Date2, -1, Step)

    for i in range(NumNumberDays):
        DaysVec = arange(Date1[i], Date2[i]+Step[i], Step[i])

        # Find weekday numbers
        DayNum = weekday(DaysVec)

        # Remove Sat and Sun
        NumberDays[i] = len(where(DayNum != 1 & DayNum != 7)) - NumberHolidays[i]

    # Build flag to change number of days between dates to negative in cases where
    # Date1 precedes Date2
    NumberDays = where(Date1 > Date2, -NumberDays, NumberDays)

    return NumberDays