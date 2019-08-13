import numpy as np


def nelson_siegel_yield(tau, theta):
    """For details, see here.

    Parameters
    ----------
        tau : array, shape (n_,)
        theta : array, shape (4,)

    Returns
    -------
        y : array, shape (n_,)

    """

    y = theta[0] - theta[1] * \
        ((1 - np.exp(-theta[3] * tau)) /
         (theta[3] * tau)) + theta[2] * \
        ((1 - np.exp(-theta[3] * tau)) /
         (theta[3] * tau) - np.exp(-theta[3] * tau))

    return np.squeeze(y)
