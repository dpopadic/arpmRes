import numpy as np
from scipy.optimize import minimize

from arpym.pricing.nelson_siegel_yield import nelson_siegel_yield


def fit_nelson_siegel_yield(tau, y, theta_0=None):
    """For details, see here.

    Parameters
    ----------
        tau : array, shape (n_,)
        y : array, shape (n_,)
        theta_0: array, shape (4,)

    Returns
    -------
        theta : array, shape (4,)

    """

    n_ = len(tau)

    def minimization_target(theta):

        # Step 1: Compute Nelson-Siegel yield curve

        y_theta = nelson_siegel_yield(tau, theta)
        output = 0.0

        # Step 2: Compute minimization function

        for n in range(n_):
            if n == 0:
                h = tau[n + 1] - tau[n]
            elif n == n_ - 1:
                h = tau[n] - tau[n - 1]
            else:
                h = tau[n + 1] - tau[n - 1]
            output += h * np.abs(y[n] - y_theta[n])

        return output

    if theta_0 is None:
        theta_0 = 0.1 * np.ones(4)

    # Step 3: Fit Nelson-Siegel parameters

    res = minimize(minimization_target, theta_0)
    theta = res.x

    return theta
