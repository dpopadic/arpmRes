from numpy import arange, zeros, var, log, exp
from numpy import sum as npsum

from FitGARCHFP import FitGARCHFP


def GarchResiduals(x,t_garch=None,p_garch=None,g=0.95,p0=[0,0.01,0.8,0]):
    # This function computes the residuals of a Garch((1,1)) fit on x.
    # If t_garch < t_obs=x.shape[1] the fit is performed on a rolling window of
    # t_garch observations
    #  INPUTS
    #  x        [matrix]: (n_ x t_obs) dataset of observations
    #  t_garch  [scalar]: number of observations processed at every iteration
    #  p_garch  [vector]: (1 x t_end) flexible probabilities (optional default=exponential decay flexible probability with half life 6 months)
    #  g        [scalar]: we impose the contraint a+b <= g on the GARCH(1,1) parameters (default: g=0.95)
    #  p0  [vector]: (1 x 4) initial guess (for compatibility with OCTAVE)
    #  OPS
    #  epsi     [matrix]: (n_ x t_end) residuals
    # note: sigma**2 is initialized with a forward exponential smoothing

    ## Code
    if t_garch is None:
        t_garch=x.shape[1]

    if p_garch is None:
        lambda1=log(2)/180
        p_garch=exp(-lambda1*arange(t_garch,1+-1,-1)).reshape(1,-1)
        p_garch=p_garch/npsum(p_garch)

    n_,t_obs=x.shape
    lam=0.7

    if t_garch==t_obs: #no rolling window
        epsi = zeros((n_,t_obs))
        for n in range(n_):
            s2_0=lam*var(x[n,:],ddof=1)+(1-lam)*npsum((lam**arange(0,t_obs-1+1))*(x[n,:]**2))
            _,_,epsi[n,:],_=FitGARCHFP(x[[n],:],s2_0,p0,g,p_garch)#GARCH fit

    else:
        t_=t_obs-t_garch #use rolling window
        epsi = zeros((n_, t_))
        for t in range(t_):
            for n in range(n_):
                x_t=x[n,t:t+t_garch-1]
                s2_0=lam*var(x_t)+(1-lam)*npsum((lam**arange(0,t_garch-1+1))*(x_t**2))
                _,_,e,_=FitGARCHFP(x_t,s2_0,p0,g,p_garch)
                epsi[n,t]=e[-1]
    return epsi

