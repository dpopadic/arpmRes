import numpy as np
from numpy import tile, r_, concatenate

from FPmeancov import FPmeancov
from SmartInverse import SmartInverse


def OrdLeastSquareFPNReg(X, Z, p, smartinverse=0):
    # This function computes the Ordinary Least Square with Flexible
    # Probabilities (OLSFP) estimator of loadings and dispersion of a
    # regression LFM.
    #  INPUTS
    #   X            :[matrix] (n_ x t_end) time-series of target variables
    #   Z            :[matrix] (k_ x t_end) time-series of factors
    #   p            :[vector] (1 x t_end) flexible probabilities
    #  smartinverse  :[scalar] additional parameter: set it to 1 to use
    #                  LRD smart inverse in the regression process
    #  OPS
    #   beta_OLSFP   :[matrix] (n_ x k_) OLSFP estimator of loadings
    #   s2_OLSFP     :[matrix] (n_ x n_) OLSFP estimator of dispersion of residuals
    #   alpha_OLSFP  :[vector] (n_ x 1) OLSFP estimator of the shifting term
    #   U            :[matrix] (n_ x t_end) time-series of fitted residuals

    # For details on the exercise, see here .
    ## code
    n_, t_ = X.shape
    k_ = Z.shape[0]
    m_XZ, s2_XZ = FPmeancov(concatenate((X, Z), axis=0), p)
    s_XZ = s2_XZ[:n_, n_:n_ + k_]
    s_ZZ = s2_XZ[n_:n_ + k_, n_:n_ + k_]

    if smartinverse == 0:
        beta_OLSFP = np.dot(s_XZ, np.linalg.pinv(s_ZZ))
    else:
        beta_OLSFP = s_XZ @ SmartInverse(s_ZZ)

    alpha_OLSFP = m_XZ[:n_] - beta_OLSFP @ m_XZ[n_:n_ + k_]

    U = X - tile(alpha_OLSFP, (1, t_)) - beta_OLSFP @ Z
    _, s2_OLSFP = FPmeancov(U, p)
    return alpha_OLSFP, beta_OLSFP, s2_OLSFP, U
