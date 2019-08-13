import numpy as np

from arpym.estimation.cointegration_fp import cointegration_fp
from arpym.estimation.fit_lfm_mlfp import fit_lfm_mlfp
import warnings


def fit_var1(x, p=None, *, nu=10**9, tol=1e-2, maxiter=500):
    """For details, see here.

    Parameters
    ----------
        x : array, shape (t_,n_)
        p : array, shape(t_, )
        nu : scalar, optional
        tol : scalar, optional
        maxiter: scalar, optional


    Returns
    -------
        b_hat : array, shape(n_,n_)
        mu_epsi_hat : array, shape(n_, )
        sigma2_hat :  array, shape(n_,n_)

    """

    t_ = x.shape[0]

    if len(x.shape) == 1:
        x = x.reshape((t_, 1))
        n_ = 1
    else:
        n_ = x.shape[1]

    if p is None:
        p = np.ones(t_) / t_

    # Step 0: find cointegrated relationships
    c_hat, _ = cointegration_fp(x, p, b_threshold=1)
    l_ = c_hat.shape[1]

    # Step 1: fit VAR(1)

    if l_ == n_:
        # Step 1a: fit stationary process
        x_t = x[1:, :]
        x_tm1 = x[:-1, :]
        mu_epsi_hat, b_hat, sig2_hat, _ = \
            fit_lfm_mlfp(x_t, x_tm1, p[1:] / np.sum(p[1:]), nu, tol=tol,
                         maxiter=maxiter)

    if l_ < n_ and l_ > 0:
        # Step 1b: fit cointegrated process
        x_t = np.diff(x, axis=0)
        x_tm1 = (x @ c_hat)[:-1, :]
        mu_epsi_hat, d_hat, sig2_hat, _ = \
            fit_lfm_mlfp(x_t, x_tm1, p[1:] / np.sum(p[1:]), nu, tol=tol,
                         maxiter=maxiter)
        if np.ndim(d_hat) < 2:
            d_hat = d_hat.reshape((-1, 1))
        b_hat = np.eye(n_) + d_hat @ c_hat.T

    if l_ == 0:
        # Step 1c: fit VAR(1) to differences
        warnings.warn('Warning: non-cointegrated series detected. ' +
                      'Fit performed on differences')

        delta_x_t = np.diff(x, axis=0)[1:]
        delta_x_tm1 = np.diff(x, axis=0)[:-1]
        mu_epsi_hat, b_hat, sig2_hat, _ = \
            fit_lfm_mlfp(delta_x_t, delta_x_tm1, p[2:] / np.sum(p[2:]), nu,
                         tol=tol, maxiter=maxiter)

        # b_hat is close to zero: detect random walk
        if np.linalg.norm(b_hat.reshape(-1, 1)) < tol:
            # random walk
            warnings.warn('Warning: a random walk has been fitted')
            b_hat = np.eye(n_)

    return np.squeeze(b_hat), np.squeeze(mu_epsi_hat), np.squeeze(sig2_hat)
