import numpy as np
from numpy import array, unique, zeros, sort, where, argsort, r_, ones
from numpy import sum as npsum

from datetime import datetime


def RawMigrationDb2AggrRiskDrivers(db,t_start,t_end):
    # This function processes the raw database of credit migrations to extract
    # the aggregate risk drivers.
    #  INPUTS
    # db       :[struct] raw database
    # t_start  :[string] time window's starting date
    # t_end    :[string] time window's ending date
    #  OPS
    # dates    :[vector] vector of dates corresponding to migrations
    # N        :[cell] N{t}[i] is the number of obligors with rating i at time dates[t]
    # n        :[cell] n{t}(i,j) is the cumulative number of transitions between ratings i and j up to time dates[t]
    # m        :[cell] m{t}(i,j) is the number of transitions occured at time dates[t] between ratings i and j
    # n_tot    :[vector] n[t] is the total number of transitions up to time dates[t]
    # fin_rat  :[cell] contains the issuers (first row) with their corresponding final ratings (second row)

    ## Code
    ratings_str = db.ratings
    rr_ = len(ratings_str)
    db.data[1] = list(map(lambda x: datetime.strptime(x, '%d-%b-%Y'), db.data[1]))

    idx_dates = (db.data[1] >= t_start) & (db.data[1] <= t_end)
    data_raw = db.data[:,idx_dates]# dataset inside time-window

    ## Transform "cell" dataset into "double"
    # issuers
    issuers_raw = data_raw[0]
    issuers_d = array(list(map(float,issuers_raw)))
    issuers = unique(issuers_d)
    s_ = len(issuers)
    # dates
    dates_raw = data_raw[1]
    dates_d = dates_raw
    dates = unique(dates_d)
    t_ = len(dates)
    # ratings
    ratings_raw = data_raw[2,:]
    ratings_d = zeros((1,len(ratings_raw)),dtype=int)
    for r in range(rr_):
        idx = ratings_str[r]==ratings_raw
        ratings_d[0,idx] = r+1

    data_d = r_[issuers_d[np.newaxis,...], dates_d[np.newaxis,...], ratings_d]# dataset in "double" format

    ## Process data
    matrix = np.NaN*ones((s_,t_),dtype=int)

    for s in range(s_):
        idx_issuer = data_d[0]==issuers[s]
        data_temp = data_d[:,idx_issuer]
        dates_temp = data_temp[1]
        dates_temp,idx_t = sort(dates_temp), argsort(dates_temp)
        data_temp = data_temp[:,idx_t]
        if len(dates_temp)==1:
            idx_1 = where(dates==dates_temp)[0][0]
            matrix[s,idx_1:] = data_temp[2]
        else:
            idx_2 = where(dates==dates_temp[-1])[0][0]
            matrix[s,idx_2:] = data_temp[2,-1]
            for t in range(1,len(dates_temp)):
                idx_1 = where(dates==dates_temp[-t-1])[0][0]
                matrix[s,idx_1:idx_2] = data_temp[2,-t-1]
                idx_2 = idx_1

    ## Compute aggregate risk drivers
    m = zeros((t_,rr_,rr_))
    n = zeros((t_,rr_,rr_))
    n_tot = zeros((1,t_))
    N = zeros((t_,rr_))
    for t in range(t_):
        for i in range(rr_):
            N[t,i] = npsum(matrix[:,t]==i+1)
            if t>0:
                for j in range(rr_):
                    if i!=j:
                        # number of transitions realized at time t between ratings i and j
                        m[t,i,j] = npsum((matrix[:,t-1]==i+1)*(matrix[:,t]==j+1))
        if t>0:
            # number of transitions, between ratings i and j, realized up to time t
            n[t] = n[t-1]+m[t]
            # total number of transitions up to time t
            n_tot[0,t] = npsum(n[t])

    ## Final ratings
    issuers_raw_ = unique(issuers_raw)

    fin_rat = {1:zeros(s_),2:{}}
    for s in range(s_):
        fin_rat[1][s] = issuers_raw_[s]
        fin_rat[2][s] = ratings_str[int(matrix[s,-1])-1]
    return dates, N, n, m, n_tot, fin_rat
