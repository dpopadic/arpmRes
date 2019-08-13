import matplotlib.pyplot as plt
from numpy import arange, sqrt
from numpy import min as npmin, max as npmax
from numpy.linalg import norm
from scipy.interpolate import interp1d
from scipy.stats import norm

plt.style.use('seaborn')


def Delta2MoneynessImplVol(sigma_delta,delta,tau,y,m_grid=None):

    # This function, given the implied volatility as a function of
    # delta-moneyness for a fixed time to maturity, computes the implied
    # volatility as a function of m-moneyness at the m_moneyness points
    # specified in m_grid.

    # INPUTS
    #  sigma_delta [vector]: (1 x k_) implied volatility as a function of
    #                           delta-moneyness
    #  delta [vector]: (1 x k_) delta-moneyness corresponding to sigma_delta
    #  tau [scalar]: time to maturity
    #  y [scalar]: risk free rate
    #  m_grid [vector]: (1 x ?) points at which sigma_m is computed
    #                           (optional: the default value is an equispaced
    #                                      grid with 100 spaces)
    # OUTPUTS
    #  sigma_m [vector]: (1 x ?) implied volatility as a function of
    #                       m-moneyness
    #  m_grid [vector]: (1 x ?) m-moneyness corresponding to sigma_m

    ## Code
    m_data = norm.ppf(delta)*sigma_delta-(y+sigma_delta**2/2)*sqrt(tau)

    if m_grid is None:
        # default option: equispaced grid with 100 spaces
        n_grid = 100
        m_grid = npmin(m_data) + (npmax(m_data)-npmin(m_data))*arange(n_grid+1)/n_grid # m-moneyness

    interp = interp1d(m_data.flatten(),sigma_delta.flatten(),fill_value='extrapolate')
    sigma_m = interp(m_grid.flatten())
    return sigma_m,m_grid



