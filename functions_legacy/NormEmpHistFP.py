import matplotlib.pyplot as plt
from numpy import arange, zeros, sqrt
from numpy import sum as npsum
from numpy.linalg import norm
from scipy.stats import norm

plt.style.use('seaborn')


def NormEmpHistFP(Epsi, p, tau, k_):
    # This function estimates the empirical histogram with Flexible
    # Probabilities of an invariant whose distribution is
    # represented in terms of simulations/historical realizations
    # INPUT
    #  Epsi  :[vector](1 x t_end) MC scenarios/historical realizations
    #  p     :[vector](1 x t_end) flexible probabilities
    #  tau   :[scalar] projection horizon
    #  k_    :[scalar] coarseness level
    # OP
    #  xi    :[1 x k_] centers of the bins
    #  f     :[1 x k_] discretized pdf of invariant

    # For details on the exercise, see here .

    ## code

    # bins width
    a = -norm.inv(10**(-15), 0, sqrt(tau))
    h = 2*a/k_

    # centers of the bins
    xi = arange(-a+h , a+ h , h )

    # frequency
    p_bin = zeros((len(xi),1))
    for k in range(len(xi)):
        index = (Epsi > xi[k] - h/2) & (Epsi <= xi[k] + h/2)
        p_bin[k] = npsum(p[index])

    # discretized pdf of an invariant
    f = 1/h * p_bin # normalized heights
    f[k_] = 1/h * (1-npsum(p_bin[:-1]))

    return xi, f
