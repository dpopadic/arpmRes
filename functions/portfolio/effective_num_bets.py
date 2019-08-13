import numpy as np


def effective_num_bets(beta, sigma2, a):
    """For details, see here.

    Parameters
    ----------
        beta : array, shape (k_+1,)
        sigma2 : array, shape (k_+1, k_+1)
        a : array, shape (k_+1, k_+1)

    Returns
    -------
        enb : array, shape (j_,)
        p : array, shape (k_+1,)

    """

    p = np.linalg.solve(a.T, beta.T)*(a@sigma2@beta.T)/(beta@sigma2@beta.T)
    p[p == 0] = 10**(-250)  # avoid log[0-1] in enb computation
    enb = np.exp(-np.sum(p*np.log(p)))
    return enb, p
