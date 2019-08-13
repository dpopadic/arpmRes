import numpy as np
from scipy.linalg import sqrtm


def minimum_torsion(s2, max_niter=10000):
    """For details, see here.

    Parameters
    ----------
        sigma : array, shape (n_, n_)

    Returns
    -------
        t : array, shape (n_, n_)

    """

    # Correlation matrix
    sigma = np.sqrt(np.diag(s2))
    C = np.diagflat(1/sigma)@s2@np.diagflat(1/sigma)
    c = sqrtm(C)  # Riccati root of C

    n_ = s2.shape[0]
    # initialize
    d = np.ones((1, n_))
    f = np.zeros(max_niter)
    for i in range(max_niter):
        U = np.diagflat(d)@c@c@np.diagflat(d)
        u = sqrtm(U)
        q = np.linalg.solve(u, np.diagflat(d)@c)
        d = np.diag(q@c)
        pi_ = np.diagflat(d)@q  # perturbation
        f[i] = np.linalg.norm(c - pi_, ord='fro')
        if i > 1 and abs(f[i] - f[i-1])/f[i]/n_ <= 10**(-8):
            f = f[:i]
            break
        elif i == max_niter and abs(f[max_niter] - f[max_niter-1]) /\
                f[max_niter]/n_ > 10**-8:
            print('number of max iterations reached: n_iter = %d' % max_niter)
    x = pi_@(np.linalg.solve(c, np.eye(n_)))
    t = np.diagflat(sigma)@x@np.diagflat(1/sigma)
    return t
