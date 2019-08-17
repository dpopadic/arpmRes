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

# # S_AggregProjection [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_AggregProjection&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-aggre-proj-vue).

# ## Prepare the environment

# +
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import zeros, diag, eye, sqrt, r_
from numpy import sum as npsum
from numpy.linalg import solve
from numpy.random import rand

from scipy.stats import t, chi2

import matplotlib.pyplot as plt

plt.style.use('seaborn')

from NormalScenarios import NormalScenarios

# parameters
m_ = 500  # number of monitoring times
j_ = 30  # number of simulations
# -

# ## Run script S_AggregProjection

from S_AggregatesEstimation import *

# ## Generate Monte Carlo projected path scenarios for each standardized cluster aggregating factor

# +
M_c1 = zeros((m_,j_))
M_c3 = zeros((m_,j_))
Zc1_tilde_proj = zeros((k_c1,m_,j_))
Zc3_tilde_proj = zeros((k_c3,m_,j_))

for m in range(m_):
    # Multivariate normal scenarios
    N_agg,_ = NormalScenarios(zeros((k_c1 + k_c3, 1)), rho2_aggr, j_)

    # Chi-squared scenarios
    M_c1[m, :] = chi2.ppf(rand(j_), k_c1)
    M_c3[m, :] = chi2.ppf(rand(j_), k_c3)

    # path scenarios
    Zc1_tilde_proj[:, m, :] = N_agg[:k_c1,:]@sqrt(diag(1 / M_c1[m, :]))

    Zc3_tilde_proj[:, m, :] = N_agg[k_c1 :k_c1 + k_c3,:]@sqrt(diag(1 / M_c3[m, :]))
# -

# ## Recover the projected paths scenarios for the standardized cluster 1

Xc1_tilde_proj =zeros((i_c1,m_,j_))
for m in range(m_):
    N_c1_res,_ = NormalScenarios(zeros((i_c1, 1)), eye(i_c1), j_)
    Xc1_tilde_proj[:, m, :] = beta_c1@Zc1_tilde_proj[:, m, :] + diag(delta2_c1)@N_c1_res@sqrt(diag(1 / M_c1[m, :]))

# ## Recover the projected paths scenarios for the standardized cluster 3

Xc3_tilde_proj =zeros((i_c3,m_,j_))
aux = e@rho2_c3@e.T
for m in range(m_):
    Z = Zc3_tilde_proj[:, m, :]
    mu_z = rho2_c3@e.T/aux@Z
    for j in range(j_):
        sig2_z = (nu_c3 + Z[:, j].T@aux@Z[:, j])*(rho2_c3 - rho2_c3@e.T / (aux)@e@rho2_c3) / (nu_c3 + k_c3)
        N_z, _ = NormalScenarios(zeros((i_c3 - k_c3, 1)), sig2_z[:i_c3 - k_c3, :i_c3 - k_c3], j_)
        Xc3_tilde_proj[:i_c3 - k_c3, m, j] = mu_z[:i_c3 - k_c3, j] + N_z[:, j] / sqrt(M_c3[m, j] / (nu_c3 + k_c3))
        Xc3_tilde_proj[i_c3 - k_c3:, m, j] = solve(e[:, i_c3 - k_c3:].T,(Z[:, j] - npsum(e[:k_c3, :i_c3 - k_c3].T * Xc3_tilde_proj[:i_c3 - k_c3, m, j])))

# ## Compute the projected path scenarios

# +
#cluster 1
Epsi_c1_hor = zeros((i_c1,m_,j_))
for i in range(i_c1):
    for m in range(m_):
        Epsi_c1_hor[i, m, :] = mu_c1_marg[i] + sqrt(sig2_c1_marg[i])*t.cdf(Xc1_tilde_proj[i, m, :], nu_c1_marg[i])

# cluster 3
Epsi_c3_hor = zeros((i_c3,m_,j_))
for i in range(i_c3):
    for m in range(m_):
        Epsi_c3_hor[i, m, :] = mu_c3_marg[i] + sqrt(sig2_c3_marg[i])*t.cdf(Xc3_tilde_proj[i, m, :], nu_c3_marg[i])

# joint scenarios
Epsi_hor = r_[Epsi_c1_hor, Epsi_c3_hor]
