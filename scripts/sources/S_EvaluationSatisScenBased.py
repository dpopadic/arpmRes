#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # S_EvaluationSatisScenBased [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_EvaluationSatisScenBased&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=EBEvalHistoricalExample).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import pi, exp, sqrt, tile, r_, maximum
from numpy import sum as npsum
from numpy.linalg import pinv

from scipy.special import erf, erfinv
from scipy.io import loadmat

import matplotlib.pyplot as plt

plt.style.use('seaborn')
np.seterr(all='ignore')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import struct_to_dict
from FPmeancov import FPmeancov
from heaviside import heaviside
from SpectralIndexSatisf import SpectralIndexSatisf
from SatisSmoothQuantile import SatisSmoothQuantile
# -

# ## Load the temporary database generated in script S_AggregationReturnScenarioBased, which contains the scenario-probability distribution of the portfolio ex-ante performance (return)

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_AggregationScenarioBased'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_AggregationScenarioBased'), squeeze_me=True)

SPX = struct_to_dict(db['SPX'])
Y_htilde = db['Y_htilde']
SPX_thor = db['SPX_thor']
htilde = db['htilde']
p = db['p']
n_ = db['n_']
j_ = db['j_']
Pi = db['Pi']
# -

# ## Compute the scenario-probaility mean and covariance of the ex-ante return by using function FPmeancov.
# ## Then, compute the expected value, the variance and the standard deviation

# +
Y_htilde = Y_htilde.reshape(1,-1)
p = p.reshape(1,-1)

[mu_Y, s2_Y] = FPmeancov(Y_htilde, p)

Satis = namedtuple('Satis', 'E_Y variance stdev mv_2 mv mv_Hess mv_new msd msvcq cq_grad ce_erf '
                            'Bulhmann_expectation Esscher_expectation')
Satisf = namedtuple('Satisf', 'mv_grad PH VaR')
Risk = namedtuple('risk', 'variance stdev mv_2 mv mv_grad mv_Hess mv_new msd msv PH VaR cq cq_grad')

# expected value
Satis.E_Y = mu_Y

# variance
Risk.variance = s2_Y
Satis.variance = -s2_Y

# standard deviation
Risk.stdev = sqrt(s2_Y)
Satis.stdev = -sqrt(s2_Y)
# -

# ## Compute the certainty-equivalent associated to an error utility function with eta=1.
# ## Then, compute the corresponding gradient and Hessian

# +
eta = 1
utility_erf =lambda x, eta: erf(x / sqrt(2*eta))  # error utility function
ce_erf = lambda exp_utility, eta: sqrt(2)*erfinv(exp_utility)  # inverse error utility function
E_utility = utility_erf(Y_htilde, eta)@p.T  # expected utility computation

# certainty-equivalent
Satis.ce_erf = ce_erf(E_utility, eta)

# gradient
utility_erf_der =lambda x, eta: (sqrt(2 / (pi*eta))*exp(-(x ** 2)) / (2*eta))  # first order derivative error utility function
num_grad = npsum(tile(p, (n_, 1))*tile(utility_erf_der(Y_htilde, eta), (n_, 1))*Pi, 1)

Satis.ce_grad = num_grad / utility_erf_der(Satis.ce_erf, eta)

# Hessian
utility_erf_der2 =lambda x, eta: -(sqrt(2 / (pi*eta))*x*exp(-x*x/(2*eta))/eta)  # second order derivative error utility function
for j in range(j_):
    num_Hess1 = npsum(p[0,j]*utility_erf_der2(Y_htilde[0,j], eta)*Pi[:,[j]]@Pi[:, [j]].T, 1)

num_Hess2 = utility_erf_der2(Satis.ce_erf, eta)*(num_grad@num_grad.T)

Satis.ce_Hess = num_Hess1 / utility_erf_der(Satis.ce_erf, eta) - num_Hess2 / (utility_erf_der(Satis.ce_erf, eta) ** 3)
# -

# ## Compute the quantile-based index of satisfaction with confidence c=0.99
# ## by implementing the smooth quantile (use function SatisSmoothQuantile)

c = 0.99  # confidence level
Satis.q, _ = SatisSmoothQuantile(1 - c, Pi, htilde, p)  # index of satisfaction
Risk.VaR = -Satis.q  # Value at Risk (risk measure)

# ## Compute the conditional quantile (spectral index) with confidence c=0.99 using function
# ## SpectralIndexSatisf, and compute also the corresponding gradient

# +
phi_cq = lambda x: (1 / (1 - c))*(heaviside(x) - heaviside(x - (1 - c)))  # spectrum

# conditional quantile
Satis.cq, _ = SpectralIndexSatisf(phi_cq, Pi, htilde, p)
Risk.es = - Satis.cq
# Expected shortfall (risk measure)

# gradient
Satis.cq_grad = (1 / (1 - c))*npsum(Pi[:, Y_htilde[0] <= Satis.q]*tile(p[[0],Y_htilde[0] <= Satis.q], (n_, 1)), 1)
# -

# ## Compute the Sharpe ratio. Then, setting the target equal to y=0.04, compute the Sortino ratio and the omega ratio

Satis.Sharpe_ratio = mu_Y / sqrt(s2_Y)
y = 0.04
Satis.Sortino_ratio = (mu_Y - y) / sqrt((maximum(y - Y_htilde, 0) ** 2)@p.T)
Satis.omega_ratio = (mu_Y - y) / (maximum(y - Y_htilde, 0)@p.T) + 1

# ## Consider as risk factor the return on the S&P 500 and compute its covariance by using
# ## function FPmeancov. Then, where the beta, the alpha and the correlation.

# +
# Z = return of the S&P index
Z = (SPX_thor.T - tile(exp(SPX.x_tnow)[...,np.newaxis], (1, j_)))/exp(SPX.x_tnow)
_, cov_YZ = FPmeancov(r_[Y_htilde, Z], p)

# beta
beta = cov_YZ[0, 1] / cov_YZ[1, 1]  # beta
Satis.beta = -beta  # beta (index of satisfaction)

# alpha
Perf_adj = Y_htilde - beta*Z  # adjusted performance
Satis.alpha = Perf_adj@p.T  # alpha

# correlation
Satis.corr = -(cov_YZ[0, 1]) / (sqrt(s2_Y * cov_YZ[1, 1]))
# -

# ## Set eta = 1 and compute the Buhlmann expectation and the Esscher expectation

zeta = 1
Satis.Bulhmann_expectation = FPmeancov((exp(-zeta*Z) * Y_htilde)/FPmeancov(exp(-zeta*Z), p)[0], p)[0]
Satis.Esscher_expectation = FPmeancov(exp(-zeta*Y_htilde) * Y_htilde, p)[0] / FPmeancov(exp(-zeta*Y_htilde), p)[0]
