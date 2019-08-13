import numpy as np
from numpy import arange, zeros, \
    sqrt, tile
from numpy import min as npmin, max as npmax
from scipy.interpolate import interp1d
from scipy.stats import norm
from Delta2MoneynessImplVol import Delta2MoneynessImplVol


def sigmadelta2mtau(n_grid_option, sigma_delta, maturity_opt, delta, tau_yield_curve, yield_curve):
    # This function implements the change of parametrization for the implied volatility
    # of an option from delta-moneyness to m-moneyness
    #  INPUTS
    #   n_grid_option      [integer]: number of points to consider in the grid for m-moneyness
    #   sigma_delta        [matrix]: implied volatility as a function of delta-moneyness
    #   maturity_opt       [matrix]: times to maturity
    #   delta              [matrix]: delta-moneyness corresponding to sigma_delta
    #   yield_curve        [matrix]: yield curve
    #
    #  OUTPUTS
    #   implVol_t          [matrix]: implied volatility m-moneyness parametrized
    #   m_grid_option      [matrix]: grid for m-moneyness
    #   t_                 [scalar]: length of time series
    #   nmat_Option        [scalar]: number of times to maturity

    t_ = sigma_delta.shape[2]
    nmat_Option = len(maturity_opt)
    k_ = len(delta)
    y_grid_t = zeros((nmat_Option, k_, t_))
    for t in range(t_):
        y_grid_tmp = interp1d(np.squeeze(tau_yield_curve), np.squeeze(yield_curve[:, t]))
        y_grid_t[:, :, t] = tile(np.atleast_2d(y_grid_tmp(maturity_opt)).T, (1, k_))
    # Compute the m-parametrized log-implied volatility surface by means of the function Delta2MoneynessImplVol, and reshape it to a 2-dimensional matrix
    # Find the maximal and the minimal m-moneyness over the observations, and build an equispaced grid between these two values
    # construct the moneyness grid

    max_m = npmax(tile(norm.ppf(delta)[np.newaxis, ..., np.newaxis], (nmat_Option, 1, t_)) * sigma_delta - (
                y_grid_t + sigma_delta ** 2 / 2) *
                  tile(sqrt(maturity_opt)[..., np.newaxis, np.newaxis], (1, k_, t_))) * 0.8
    min_m = npmin(tile(norm.ppf(delta)[np.newaxis, ..., np.newaxis], (nmat_Option, 1, t_)) * sigma_delta - (
                y_grid_t + sigma_delta ** 2 / 2) *
                  tile(sqrt(maturity_opt)[..., np.newaxis, np.newaxis], (1, k_, t_))) * 0.8

    m_grid_option = min_m + (max_m - min_m) * arange(n_grid_option + 1) / n_grid_option

    # For each observation, use function Delta2MoneynessImplVol to pass from the delta-parametrized to the m-parametrized implied volatility surface
    implVol_t = zeros((nmat_Option, n_grid_option + 1, t_))
    for t in range(t_):
        for n in range(nmat_Option):
            implVol_t[n, :, t] = \
            Delta2MoneynessImplVol(sigma_delta[[n], :, t], delta, maturity_opt[[n]], y_grid_t[n, :, [t]],
                                   m_grid_option)[0]  # linear interpolation

    return implVol_t, m_grid_option, t_, nmat_Option
