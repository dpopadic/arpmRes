import numpy as np
from scipy.stats import norm


def cornish_fisher(mu, sd, sk, c=None):
    """For details, see here.

    Parameters
    ----------
        mu : scalar
        sd : scalar
        sk : scalar
        c : array, shape(arbitrary length, )

    Returns
    -------
        q : scalar

    """

    if c is None:
        c = np.arange(.001, 1, 0.001)
    z = norm.ppf(c)

    # Cornish-Fisher expansion
    q = mu + sd*(z + sk / 6 * (z**2 - 1))

    return q
