import numpy as np
from numpy import diag, eye, round, diagflat, real
from numpy import int as int
from numpy.linalg import inv as inv
from numpy.linalg import cond as cond
from numpy.linalg import solve as solve

EPS = np.spacing(1)

from functions_legacy.FactorAnalysis import FactorAnalysis


def SmartInverse(sig2, tol=EPS):
    # This function computes an approximating inverse using factor analysis if
    # the input matrix is badly conditioned (tol=eps by default)
    # INPUT
    #  sig[1:]     [matrix] (n_ x n_) positive definite matrix
    # OP
    #  sig2_inv: [matrix] (n_ x n_) approximate inverse of sig2 computed using
    #                               factor analysis
    ## Code
    rcond = 1 / cond(sig2)
    if rcond < tol:
        n_ = sig2.shape[0]
        k_ = int(max(1, round(
            0.1 * n_)))  # We empirically choose a number of factors equal to about 10% of the dimension of sig2

        # Step 1. Correlation matrix
        sig2_vol_inv = diagflat(diag(sig2) ** (-1 / 2))
        rho2 = sig2_vol_inv @ sig2 @ sig2_vol_inv

        # Step 2. Factor analysis
        _, beta, _, _, _ = FactorAnalysis(rho2, np.array([[0]]), k_)
        delta2 = diag(eye((n_)) - beta @ beta.T)  # Residual vector

        # Step 3. (Approximated) inverse of correlation matrix
        omega2 = diag(1 / delta2)
        rho2_inv = omega2 - omega2 @ beta @ solve(beta.T @ omega2 @ beta + eye((k_)), beta.T) @ omega2

        # Step 4. (Approximated) inverse of sig2
        sig2_inv = sig2_vol_inv @ rho2_inv @ sig2_vol_inv
    else:
        sig2_inv = inv(sig2)

    sig2_inv = (sig2_inv + sig2_inv.T) / 2

    return sig2_inv
