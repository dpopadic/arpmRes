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

# # S_PortfolioOptimization [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_PortfolioOptimization&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=PortfolioOptimLRD).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

from numpy.ma import array

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import arange, diag, eye, log, exp, sqrt, tile
from numpy.linalg import solve, inv, pinv, norm

from scipy.io import loadmat

import matplotlib.pyplot as plt

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import struct_to_dict
from FPmeancov import FPmeancov
from FactorAnalysis import FactorAnalysis

# inputs
n_ = 300
k_ = 5  # number of factors
s2_Z = eye(k_)
r = 0.02  # risk-free rate
a_p = 1000  # excess performance
# -

# ## Upload database

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_StocksS_P'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_StocksS_P'), squeeze_me=True)  # stock data

Data = struct_to_dict(db['Data'])
# -

# ## Compute the historical daily realization of the compounded returns (invariants), which become the scenarios for the compounded returns C_{t->t+1}

# +
v_historical = Data.Prices
C = log(v_historical[:n_, 1:])-log(v_historical[:n_, : -1])

n_, j_ = C.shape
# -

# ## Set the historical Flexible Probabilities as exponential decay with half life 2 years

lam = log(2) / 500  # half life 2y
p = exp(-lam*arange(j_, 1 + -1, -1)).reshape(1,-1)

# ## Projection: compute the reconstructed scenarios for the risk drivers at the horizon
# ##note: no projection is needed as the estimation step coincides with the
# ##time to horizon, i.e. u=t+1

# +
# current prices
v_t = Data.Prices[:n_, [-1]]

X_u = log(tile(v_t, (1, j_))) + C
# -

# ## Pricing: compute the scenarios for the P&L of each stock by full repricing
# ## scenarios for prices tomorrow

# +
V_u = exp(X_u)

# P&L's scenarios
Pi = V_u - tile(v_t, (1, j_))
# -

# ## Compute HFP-covariance

m_Pi_HFP, s2_Pi_HFP = FPmeancov(Pi, p)
s_Pi_HFP = sqrt(diag(s2_Pi_HFP))

# ## Compute the optimal portfolio with the HFP-covariance of the P&L's

# +
a = m_Pi_HFP - r*v_t  # instruments' excess performance

# compute the inverse of s2_Pi

inv_s2_Pi_HFP = solve(s2_Pi_HFP,eye(s2_Pi_HFP.shape[0]))
# t_HFP = toc

# compute optimal portfolio with HFP covariance
h_star_HFP = a_p*(inv_s2_Pi_HFP@a) / (a.T@inv_s2_Pi_HFP@a)
# -

# ## Perform factor analysis on P&L's correlation matrix

c2_Pi_HFP = np.diagflat(1 / s_Pi_HFP)@s2_Pi_HFP@np.diagflat(1 / s_Pi_HFP)
_, beta_tilde,*_ = FactorAnalysis(c2_Pi_HFP, array([[0]]), k_)

# ## deduce low-rank-diagonal decomposition of s2_Pi:  s2_Pi = beta@s2_Z@beta.T + diag(diag_s2_U)

# +
beta = np.diagflat(s_Pi_HFP)@beta_tilde
diag_s2_U = diag(s2_Pi_HFP) * diag(eye(n_) - beta_tilde@beta_tilde.T)

# reconstruct the low-rank-diagonal covariance
s2_Pi_lrd = beta@s2_Z@beta.T + diag(diag_s2_U)
# -

# ## Compute optimal portfolio composition with low-rank-diagonal covariance
# ## compute the inverse of s2_Pi_lrd

# +
# tic
omega2 = diag(1 / diag_s2_U)
inv_s2_Pi_lrd = omega2 - (omega2@beta).dot(pinv((beta.T@omega2@beta + inv(s2_Z))))@beta.T@omega2
# t_lrd = toc

# compute optimal portfolio with low-rank-diagonal covariance
h_star_lrd = a_p*(inv_s2_Pi_lrd@a) / (a.T@inv_s2_Pi_lrd@a)
# -

# ## Compute the distance between portfolios

delta_h = norm((h_star_lrd - h_star_HFP) * v_t) / norm(((h_star_lrd + h_star_HFP) / 2) * v_t)
