from numpy import sum as npsum
from numpy import zeros, sort, cumsum, sqrt
from numpy.random import rand, randn

from scipy.stats import poisson


def JumpDiffusionMerton(mu,sigma,lam,mu_jump,sigma_jump,ts,j_):
    # Simulate a jump diffusion process
    #  INPUTS
    # mu          :[scalar] mean parameter of Gaussian distribution
    # sigma       :[scalar] standard deviation of Gaussian distribution
    # lam      :[scalar] Poisson intensity rate of jumps
    # mu_jump     :[scalar] drift of log-jump
    # sigma_jump  :[scalar] st.dev of log-jump
    # ts          :[vector] time steps, with ts[0]=0
    # j_          :[scalar] number of simulations
    #  OPS
    # x           :[matrix](j_ x len(ts)) matrix of simulated paths

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
        S=mu_jump+sigma_jump*randn(n_jump[j],1)

        # put things together
        CumS=cumsum(S)
        for k in range(1,k_):
            events=npsum(t<=ts[k])
            if events:
                jumps[j,k] = CumS[events-1]

    #simulate the Brownian motion component
    d_BM = zeros((j_,k_))
    for k in range(1,k_):
        dt = ts[k]-ts[k-1]
        d_BM[:,[k]] = mu*dt + sigma*sqrt(dt)*randn(j_,1)

    #put together the arithmetic BM with the Poisson jumps
    x = cumsum(d_BM,1)+jumps
    return x
