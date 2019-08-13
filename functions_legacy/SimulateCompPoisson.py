from numpy import sum as npsum
from numpy import zeros, sort, cumsum
from numpy.random import rand
from scipy.stats import poisson, expon

from FPmeancov import FPmeancov
from HFPquantile import HFPquantile


def SimulateCompPoisson(lam, jumps_ts, p, ts, j_, method):
    # Simulate a Compound Poisson Process
    #  INPUTS
    # lam   :[scalar] Poisson process arrival rate
    # jumps_ts :[vector](1 x t_end) time series of realized jumps
    # p        :[vector](1 x t_end) vector of Flexible Probabilities associated with jumps_ts
    # ts       :[row vector] vector of future time steps with ts[0]=0
    # j_       :[scalar] number of simulations
    # method   :[string] ExpJumps or FromHistogram, chooses how to model jumps
    #  OPS
    # x        :[matrix](j_ x len(ts)) simulated paths

    ## Code
    tau = ts[0,-1]
    k_ = ts.shape[1]
    # simulate number of jumps
    n_jumps = poisson.rvs(lam*tau, size=(j_, 1))

    jumps = zeros((j_,k_))
    for j in range(j_):
        # simulate jump arrival time
        t = tau*rand(1,n_jumps[j,0])
        t = sort(t)

        # simulate jumps size
        if method == 'FromHistogram':
            c = rand(1,n_jumps[j,0])
            S = HFPquantile(jumps_ts,c,p)
#             for k in range(n_jumps([j])
#                 S[k] = HFPquantile((jumps_ts,c[k],p))
#
        elif method=='ExpJumps':
            #fit of the exponential parameter with FP
            mu, _ = FPmeancov(jumps_ts,p)
            S = expon.rvs(scale=mu,size=(1,n_jumps[j,0]))

        # put things together
        CumS = cumsum(S)
        for k in range(1,k_):
            events = npsum(t <= ts[0,k])
            if events>1:
                jumps[j,k] = CumS[events-1]

    x = jumps #[zeros(j_, k_) + jumps]
    return x
