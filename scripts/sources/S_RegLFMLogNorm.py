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

# # S_RegLFMLogNorm [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_RegLFMLogNorm&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-fact-demand-horiz-eff).

# ## Prepare the environment

# +
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import array, ones, diag, exp
from numpy.linalg import pinv, norm

import matplotlib.pyplot as plt

plt.style.use('seaborn')

from NormalScenarios import NormalScenarios
from OrdLeastSquareFPNReg import OrdLeastSquareFPNReg

# inputs
n_ = 3  # number of stocks
k_ = 2  # number of factors
j_ = 30000  # number of simulations

mu_YF = array([[0.05, 0.18, -0.23, 0.09, 0.03]]).T  # mean vector

sig2_YF = array([[0.50, - 0.05, - 0.30, - 0.18, - 0.51],
           [ -0.05,  0.55, - 0.02, - 0.29, - 0.16],
           [- 0.30, - 0.02,  0.52,  0.27,  0.45],
           [- 0.18, - 0.29,  0.27,  0.51,  0.37],
           [- 0.51, - 0.16,  0.45,  0.37,  0.66]])  # joint covariance
# -

# ## Compute LFM parameters analytically

# +
mu_Y = mu_YF[:n_]
mu_F = mu_YF[n_:n_+ k_]

sig2_Y = sig2_YF[:n_, :n_]
sig_YF = sig2_YF[:n_, n_ :n_+ k_]
sig2_F = sig2_YF[n_ :n_+ k_, n_ :n_ + k_]

# computation of beta
exp_Y = exp(mu_Y + diag(sig2_Y).reshape(-1,1) / 2)
exp_F = exp(mu_F + diag(sig2_F).reshape(-1,1) / 2)
beta = np.diagflat(exp_Y)@(exp(sig_YF) - ones((n_, k_))).dot(pinv((exp(sig2_F) - ones((k_, k_)))@np.diagflat(exp_F)))

# computation of alpha
alpha = exp_Y - ones((n_, 1)) - beta@(exp_F - ones((k_, 1)))
# -

# ## Generate simulations for variables Y,F and deduce simulations for X,Z

# +
YF = NormalScenarios(mu_YF, sig2_YF, j_, 'Riccati')[0]

XZ = exp(YF) - 1
X = XZ[:n_,:]
Z = XZ[n_:n_ + k_,:]
# -

# ## Set Flexible Probabilities

p = ones((j_, 1)) / j_

# ## Estimate regression LFM

[alpha_OLSFP, beta_OLSFP, s2_OLSFP, U] = OrdLeastSquareFPNReg(X, Z, p)

# ## Compute estimation errors

er_alpha = norm(alpha - alpha_OLSFP)
er_beta = norm(beta - beta_OLSFP)
