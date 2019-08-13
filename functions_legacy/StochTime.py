from numpy import abs, sqrt
from scipy.stats import gamma


def StochTime(dt, process, params):
    # This function computes stochastic time increments for either VG process
    # or Heston model, depending on inputs.
    #  INPUTS
    # dt             :[scalar] incremental time step
    # process        :[string] string switch process:
    #                          - VG for Variance gamma process
    #                          - Heston for Heston model
    # params  :[struct] input parameters
    #  OPS
    # dT_t           :[column vector] stochastic time increments

    ## Code
    if process=='VG':
        nu = params.nu
        J = params.J
        # iid stochastic time increment, i.e. T_t is a subordinator
        dT_t = gamma.rvs(dt/nu, scale=nu,size=(J,1))

    elif process=='Heston':
        kappa = params.kappa
        s2_ = params.s2_
        eta = params.eta
        S2_t = params.S2_t
        z = params.z

        # square-root CIR process for the variance
        dS2_t = -kappa*(S2_t - s2_)*dt + eta*sqrt(dt*S2_t)*z
        Y_t_dt = abs(S2_t + dS2_t) # reflection scheme
        # AR[0] stochastic time increment, i.e. T_t is a Time Change
        # process
        dT_t = Y_t_dt*dt

    return dT_t
