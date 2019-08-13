from numpy import sum as npsum
from numpy import zeros, sort, cumsum, sqrt
from numpy.random import rand, randn
from scipy.stats import expon, poisson, binom


def JumpDiffusionKou(mu,sigma,lam,p,e1,e2,ts,j_):
    # Simulate a double-exponential process
    #  INPUTS
    # mu      :[scalar] mean parameter of Gausian distribution
    # sigma   :[scalar] standard deviation of Gaussian distribution
    # lam  :[scalar] Poisson intensity of jumps
    # p       :[scalar] binomial parameter of jumps
    # e1      :[scalar] exponential parameter for the up-jumps
    # e2      :[scalar] exponential parameter for the down-jumps
    # ts      :[vector] time steps with ts[0]=0
    # j_      :[scalar] number of simulations
    #  OPS
    # x       :[matrix](j_ x len(ts)) matrix of simulated paths

    ## Code

    tau=ts[-1]
    # simulate number of jumps
    n_jump=poisson.rvs(lam*tau,size=(j_))

    k_=len(ts)
    jumps=zeros((j_,k_))
    for j in range(j_):
        # simulate jump arrival time
        t=tau*rand(n_jump[j],1)
        t=sort(t)
        # simulate jump size
        ww=binom.rvs(1,p,size=(n_jump[j],1))

        S=ww*expon.rvs(scale=e1,size=(n_jump[j],1))-(1-ww)*expon.rvs(scale=e2,size=(n_jump[j],1))

        # put things together
        CumS=cumsum(S)
        for k in range(1,k_):
            events=npsum(t<=ts[k])
            if events:
                jumps[j,k]=CumS[events-1]

    #simulate the arithmetic Brownian motion component
    d_BM = zeros((j_,k_))
    for k in range(1,k_):
        dt=ts[k]-ts[k-1]
        d_BM[:,[k]]=mu*dt + sigma*sqrt(dt)*randn(j_,1)

    #put together the arithmetic BM with the jumps
    x = cumsum(d_BM,1)+jumps
    return x
