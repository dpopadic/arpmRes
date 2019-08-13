from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from numpy import sum as npsum
from numpy import zeros, diag, eye, sqrt, tile
from numpy.linalg import solve, cholesky
from numpy.random import rand
from scipy.stats import chi2, f

plt.style.use('seaborn')

from DimRedScenariosNormal import DimRedScenariosNormal
from NormalScenarios import NormalScenarios


def Tscenarios(nu, mu, sig2, j_, optionT=None, method='Riccati', d=None):
    # This function generates student t simulations whose
    # moments match the theoretical moments mu_, nu/(nu-2)@sigma2_, either from
    # radial or stochastic representation and through dimension reduction.
    #  INPUTS
    #   nu              : [scalar] degrees of freedom
    #   mu              : [vector] (n_ x 1) vector of means
    #   sigma2          : [matrix] (n_ x n_) dispersion matrix
    #   j_              : [scalar] (even) number of simulations
    #   optionT         : [struct] with fields (defaults values are 0 for both fields)
    #   optionT.dim_red : [scalar] number of factors to be used for normal
    #                     scenario generation with dimension reduction. If it is set to 0, normal
    #                     scenarios are generated without dimension reduction.
    #   optionT.stoc_rep : [scalar] Set it to 1 to generate t scenarios through
    #                     stochastic representation via normal and chi-square scenarios.
    #   method          : [string] Riccati (default), CPCA, PCA, LDL-Cholesky,
    #                              Gram-Schmidt, Chol
    #   d               : [matrix] (k_ x n_) full rank constraints matrix for CPCA
    #  OPS
    #   X               : [matrix] (n_ x j_) matrix of scenarios drawn from a
    #                     Student t distribution t(nu,mu,sig2).
    #
    #
    # NOTE: Use always a large number of simulations j_ >> n_ to ensure that
    #       NormalScenarios works properly. Also we reccommend a low number of
    #       factors k_<< n_

    # For details on the exercise, see here .
    ## Code

    if optionT is None:
        optionT = namedtuple('option', ['dim_red', 'stoc_rep'])
        optionT.dim_red = 0
        optionT.stoc_rep = 0
    n_ = len(mu)
    k_ = optionT.dim_red

    if optionT.stoc_rep == 0:
        # Step 1. Radial scenarios
        R = sqrt(n_ * f.ppf(rand(j_, 1), n_, nu))

    # Step 2. Correlation
    rho2 = np.diagflat(diag(sig2) ** (-1 / 2)) @ sig2 @ np.diagflat(diag(sig2) ** (-1 / 2))

    # Step 3. Normal scenarios
    if optionT.dim_red > 0:
        N, beta = DimRedScenariosNormal(zeros((n_, 1)), rho2, k_, j_, method, d)
    else:
        N, _ = NormalScenarios(zeros((n_, 1)), rho2, j_, method, d)

    # Step 4. Inverse
    if optionT.dim_red > 0:
        delta2 = diag(eye(n_) - beta @ beta.T)
        omega2 = np.diagflat(1 / delta2)
        rho2_inv = omega2 - omega2 @ beta / (beta.T @ omega2 @ beta + eye((k_))) @ beta.T @ omega2
    else:
        rho2_inv = solve(rho2, eye(rho2.shape[0]))

    # Step 5. Cholesky
    rho_inv = cholesky(rho2_inv)

    # Step 6. Normalizer
    M = sqrt(npsum((rho_inv @ N) ** 2, axis=0))

    # Step 7. Output
    if optionT.stoc_rep == 0:
        # Elliptical representation
        X = tile(mu, (1, j_)) + np.diagflat(sqrt(diag(sig2))) @ N @ np.diagflat(1 / M) @ np.diagflat(R)
    else:
        # Stochastic representation
        v = chi2.ppf(rand(j_, 1), nu) / nu
        X = tile(mu, (1, j_)) + np.diagflat(sqrt(diag(sig2))) @ N @ np.diagflat(sqrt((1 / v)))
    return X
