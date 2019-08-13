from collections import namedtuple

from numpy import arange, cumsum, diff
from numpy.fft import ifft, fft


from HistogramFP import HistogramFP
from DiscretizeNormalizeParam import DiscretizeNormalizeParam


def ProjDFFT(Epsi, p, tau, k_, model=None, par=None):
    # This function discretizes the pdf of an invariant and projects
    # it to future horizons via DFFT
    # INPUT
    #  Epsi  :[vector](1 x t_end) MC scenarios/historical realizations
    #  p     :[vector](1 x t_end) flexible probabilities
    #  tau   :[scalar] projection horizon
    #  k_    :[scalar] coarseness level
    #  model  :[string] specifies the distribution: shiftedLN,.TStudent t.T,Uniform
    #  par    :[struct] model parameters
    # OP
    #  xi    :[1 x k_] centers of the bins
    #  f_tau :[1 x k_] discretized pdf of invariant
    #  F_tau :[1 x k_] discretized cdf of invariant

    # For details on the exercise, see here .

    ## code

    if par is None:
        # compute the one-step empirical discretized pdf and the centers of the bins
        option = namedtuple('option',['tau','k_'])
        option.tau = tau
        option.k_ = k_
        f, xi = HistogramFP(Epsi, p, option)
    elif Epsi is None or p is None:
        # compute the one-step parametric discretized pdf
        xi, f = DiscretizeNormalizeParam(tau, k_, model, par)

    # discretized cf
    phi_hat = fft(f)

    # projected discretized cf
    h = diff(xi[:2],1) # bins width
    signs = (-1)**(arange(k_)*(tau - 1))
    phi_tau = signs*h**(tau - 1)*(phi_hat**tau)

    # projected discretized pdf of an invariant
    f_tau = ifft(phi_tau)
    F_tau = h*cumsum(f_tau)
    return xi, f_tau, F_tau
