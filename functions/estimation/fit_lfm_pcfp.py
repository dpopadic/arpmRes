import numpy as np

from arpym.statistics.meancov_sp import meancov_sp
from arpym.tools.pca_cov import pca_cov


def fit_lfm_pcfp(x, p, sig2, k_):
    """For details, see here.

    Parameters
    ----------
        x : array, shape (t_, n_)
        p : array, shape (t_,)
        sig2 : array, shape (t_, t_)
        k_ : scalar

    Returns
    -------
        alpha_PC : array, shape (n_,)
        beta_PC : array, shape (n_, k_)
        gamma_PC : array, shape(n_, k_)
        s2_PC : array, shape(n_, n_)

    """

    t_, n_ = x.shape

    # Step 1: Compute HFP-expectation and covariance of x
    m_x, s2_x = meancov_sp(x, p)

    # Step 2: Compute the Choleski root of sig2
    sig = np.linalg.cholesky(sig2)

    # Step 3: Perform spectral decomposition
    s2_tmp = np.linalg.solve(sig, (s2_x.dot(np.linalg.pinv(sig))))
    e, lambda2 = pca_cov(s2_tmp)

    # Step 4: Compute optimal loadings for PC LFM
    beta_PC = sig@e[:, :k_]

    # Step 5: Compute factor extraction matrix for PC LFM
    gamma_PC = (np.linalg.solve(sig, np.eye(n_)))@e[:, :k_]

    # Step 6: Compute shifting term for PC LFM
    alpha_PC = (np.eye(n_)-beta_PC@gamma_PC.T)@m_x

    # Step 7: Compute the covariance of residuals
    s2_PC = sig@e[:, k_:n_]*lambda2[k_:n_]@e[:, k_:n_].T@sig.T
    return alpha_PC, beta_PC, gamma_PC, s2_PC
