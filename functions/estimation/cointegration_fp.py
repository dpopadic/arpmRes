import numpy as np

from arpym.statistics.meancov_sp import meancov_sp
from arpym.estimation.fit_lfm_mlfp import fit_lfm_mlfp
from arpym.tools.pca_cov import pca_cov


def cointegration_fp(x, p=None, *, b_threshold=0.99):
    """For details, see here.

    Parameters
    ----------
         x : array, shape(t_, d_)
         p : array, shape(t_, d_)
         b_threshold : scalar

    Returns
    -------
        c_hat : array, shape(d_, l_)
        beta_hat : array, shape(l_, )

    """

    t_ = x.shape[0]
    if len(x.shape) == 1:
        x = x.reshape((t_, 1))
        d_ = 1
    else:
        d_ = x.shape[1]

    if p is None:
        p = np.ones(t_) / t_

    if p is None:
        p = np.ones(t_)/t_

    # Step 1: estimate HFP covariance matrix

    _, sigma2_hat = meancov_sp(x, p)

    # Step 2: find eigenvectors

    e_hat, _ = pca_cov(sigma2_hat)

    # Step 3: detect cointegration vectors

    c_hat = []
    b_hat = []
    p = p[:-1]

    for d in np.arange(0, d_):

        # Step 4: Define series

        y_t = e_hat[:, d] @ x.T

        # Step 5: fit AR(1)

        yt = y_t[1:].reshape((-1, 1))
        ytm1 = y_t[:-1].reshape((-1, 1))
        _, b, _, _ = fit_lfm_mlfp(yt, ytm1, p / np.sum(p))
        if np.ndim(b) < 2:
            b = np.array(b).reshape(-1, 1)

        # Step 6: check stationarity

        if abs(b[0, 0]) <= b_threshold:
            c_hat.append(list(e_hat[:, d]))
            b_hat.append(b[0, 0])

    # Output

    c_hat = np.array(c_hat).T
    b_hat = np.array(b_hat)

    # Step 7: Sort according to the AR(1) parameters beta_hat

    c_hat = c_hat[:, np.argsort(b_hat)]
    b_hat = np.sort(b_hat)

    return c_hat, b_hat
