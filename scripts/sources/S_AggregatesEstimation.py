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

# # S_AggregatesEstimation [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_AggregatesEstimation&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-aggr-cond-fac-est-vue).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../functions-legacy'))
os.getcwd()
from collections import namedtuple

import numpy as np
from numpy import arange, zeros, where, argsort, diag, eye, abs, log, exp, sqrt, tile, r_, maximum, array, diagflat, \
    diff
from numpy import sum as npsum
from numpy.linalg import pinv

np.seterr(all="ignore")

from scipy.stats import t
from scipy.io import loadmat

import matplotlib.pyplot as plt

plt.style.use('seaborn')

# relative paths not working..
from functions_legacy.ARPM_utils import struct_to_dict
from CONFIG import GLOBAL_DB, TEMPORARY_DB
from functions_legacy.MaxLikelihoodFPLocDispT import MaxLikelihoodFPLocDispT
from functions_legacy.ConditionalFP import ConditionalFP
from functions_legacy.DiffLengthMLFP import DiffLengthMLFP

from functions_legacy.FactorAnalysis import FactorAnalysis
from functions_legacy.pcacov import pcacov
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=S_AggregatesEstimation-parameters)

# +
tauHL_smoo = 30  # half-life time for smoothing
tauHL_scor = 100  # half-life time for scoring

alpha = 0.25
tauHL_prior = 21 * 4  # parameters for Flexible Probabilities conditioned on VIX

nu_vec = range(2, 31)
nu_ = len(nu_vec)

nu_c1 = 12
nu_c3 = 20
nu_aggr = 5

k_c1 = 4
k_c3 = 1
# -

# ## Upload database

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_Aggregates'))
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_Aggregates'))
try:
    dbvix = loadmat(os.path.join(GLOBAL_DB, 'db_VIX'))
except FileNotFoundError:
    dbvix = loadmat(os.path.join(TEMPORARY_DB, 'db_VIX'))

epsi_c1 = db['epsi_c1']
epsi_c3 = db['epsi_c3']
dates = db['dates']

VIX = struct_to_dict(dbvix['VIX'])
# -

# ## Compute the time series of the conditioning variable by applying sequentially smoothing and scoring filters to the time series of VIX's compounded returns

# +
c_VIX = diff(log(VIX.value)).reshape(1,-1)
t_vix = c_VIX.shape[1]
times = range(t_vix)

# smoothing
z_vix = zeros((1, t_vix))
for it in range(t_vix):
    p_smoo_t = exp(-log(2) / tauHL_smoo * (tile(it + 1, (1, it + 1)) - times[:it + 1]))
    gamma_t = npsum(p_smoo_t)
    z_vix[0, it] = npsum(p_smoo_t * c_VIX[0, :it + 1]) / gamma_t

# scoring
mu_hat = zeros((1, t_vix))
mu2_hat = zeros((1, t_vix))
sd_hat = zeros((1, t_vix))
for it in range(t_vix):
    p_scor_t = exp(-log(2)/ tauHL_scor*(tile(it+1, (1, it+1)) - times[:it+1]))
    gamma_scor_t = npsum(p_scor_t)
    mu_hat[0, it] = npsum(p_scor_t * z_vix[0, :it+1]) / gamma_scor_t
    mu2_hat[0, it] = npsum(p_scor_t * (z_vix[0, :it+1]) ** 2) / gamma_scor_t
    sd_hat[0, it] = sqrt(mu2_hat[0, it] - (mu_hat[0, it]) ** 2)

z_vix = (z_vix - mu_hat) / sd_hat
VIXdate = VIX.Date
# -

# ## Intersect the time series of invariants with the time series of the conditioning variable

# +
inter = np.in1d(VIXdate, dates[0])
_, unique = np.unique(VIXdate[0, inter], return_index=True)
indices = np.array(range(len(VIXdate[0])))[inter]
tau_vix = indices[unique]
inter = np.in1d(dates, VIXdate)
_, unique = np.unique(dates[0, inter], return_index=True)
indices = np.array(range(len(dates[0])))[inter]
tau_epsi = indices[unique]

z_vix = z_vix[0, tau_vix]
epsi_c1 = epsi_c1[:, tau_epsi]
epsi_c3 = epsi_c3[:, tau_epsi]
i_c1, _ = epsi_c1.shape
i_c3, t_ = epsi_c3.shape
# -

# ## Compute the state and time conditioning probabilities

# +
z_vix_star = z_vix[-1]  # target value
prior = exp(-(log(2) / tauHL_prior) * abs(arange(t_, 1 + -1, -1))).reshape(1,-1)
prior = prior / npsum(prior)

# conditioner
conditioner = namedtuple('conditioner', ['Series', 'TargetValue', 'Leeway'])
conditioner.Series = z_vix.reshape(1, -1)
conditioner.TargetValue = z_vix_star.reshape(1, -1)
conditioner.Leeway = alpha

p = ConditionalFP(conditioner, prior)
# -

# ## Estimate the t copula of each cluster

# +
# estimate marginal distributions by fitting a Student t distribution via
# MLFP and recover the invariants' grades

# cluster 1
u1 = zeros((i_c1, t_))
nu_c1_marg = zeros(i_c1)
mu_c1_marg = zeros(i_c1)
sig2_c1_marg = zeros(i_c1)
for i in range(i_c1):
    mu_nu = zeros(nu_)
    sig2_nu = zeros(nu_)
    like_nu = zeros(nu_)
    for k in range(nu_):
        nu_k = nu_vec[k]
        mu_nu[k], sig2_nu[k], _ = MaxLikelihoodFPLocDispT(epsi_c1[[i], :], p, nu_k, 10 ** -6, 1)
        epsi_t = (epsi_c1[i, :] - mu_nu[k]) / sqrt(sig2_nu[k])
        like_nu[k] = npsum(p * log(t.pdf(epsi_t, nu_k) / sqrt(sig2_nu[k])))  # likelihood
        j_nu = argsort(like_nu)[::-1]

    # take as estimates the parameters giving rise to the highest likelihood
    nu_c1_marg[i] = max(nu_vec[j_nu[0]], 10)
    mu_c1_marg[i] = mu_nu[j_nu[0]]
    sig2_c1_marg[i] = sig2_nu[j_nu[0]]

# cluster 3
u3 = zeros((i_c3, t_))
nu_c3_marg = zeros(i_c3)
mu_c3_marg = zeros(i_c3)
sig2_c3_marg = zeros(i_c3)
for i in range(i_c3):
    mu_nu = zeros(nu_)
    sig2_nu = zeros(nu_)
    like_nu = zeros(nu_)
    for k in range(nu_):
        nu_k = nu_vec[k]
        idx = where(~np.isnan(epsi_c3[0]))[0][0]
        p_k = p[0,idx:] / npsum(p[0,idx:])
        mu_nu[k], sig2_nu[k], _ = MaxLikelihoodFPLocDispT(epsi_c3[[i], idx:], p_k, nu_k, 10 ** -6, 1)
        epsi_t = (epsi_c3[i, idx:] - mu_nu[k]) / sqrt(sig2_nu[k])
        like_nu[k] = npsum(p_k * log(t.pdf(epsi_t, nu_k) / sqrt(sig2_nu[k])))  # likelihood
        j_nu = argsort(like_nu)[::-1]

    # take as estimates the parameters giving rise to the highest likelihood
    nu_c3_marg[i] = maximum(nu_vec[j_nu[0]], 10)
    mu_c3_marg[i] = mu_nu[j_nu[0]]
    sig2_c3_marg[i] = sig2_nu[j_nu[0]]

# Map the grades into standard Student t realizations

# cluster 1
epsi_c1_tilde = zeros((i_c1, t_))
for i in range(i_c1):
    u1[i, :] = t.cdf((epsi_c1[i, :] - mu_c1_marg[i]) / sqrt(sig2_c1_marg[i]), nu_c1_marg[i])
    epsi_c1_tilde[i, :] = t.ppf(u1[i, :], nu_c1)

# cluster 3
epsi_c3_tilde = zeros((i_c3, t_))
for i in range(i_c3):
    u3[i, :] = t.cdf((epsi_c3[i, :] - mu_c3_marg[i]) / sqrt(sig2_c3_marg[i]), nu_c3_marg[i])
    epsi_c3_tilde[i, :] = t.ppf(u3[i, :], nu_c3)

# fit the ellipsoid via MLFP

# cluster 1
_, sigma2,_ = MaxLikelihoodFPLocDispT(epsi_c1_tilde, p, nu_c1, 10 ** -6, 1)
rho2_c1 = np.diagflat(diag(sigma2) ** (-1 / 2))@sigma2@np.diagflat(diag(sigma2) ** (-1 / 2))

# cluster 3
_, sigma2 = DiffLengthMLFP(epsi_c3_tilde, p, nu_c3, 10**-6)
rho2_c3 = np.diagflat(diag(sigma2) ** (-1 / 2))@sigma2@np.diagflat(diag(sigma2) ** (-1 / 2))
# -

# ## Compute the time series of the cluster 1 aggregating variable

# +
z_tilde_c1 = zeros((i_c1, t_))
# factor analysis
rho2_c1_LRD, beta_c1, *_ = FactorAnalysis(rho2_c1, array([[0]]), k_c1)
beta_c1 = np.real(beta_c1)

# inverse LRD correlation
delta2_c1 = diag(eye((i_c1)) - beta_c1@beta_c1.T)
omega2_c1 = diagflat(1 / delta2_c1)
rho2_c1_inv = omega2_c1 - (omega2_c1@beta_c1).dot(pinv((beta_c1.T@omega2_c1@beta_c1 + eye(k_c1))))@beta_c1.T@omega2_c1

# time series aggregating variable
z_tilde_c1 = beta_c1.T@rho2_c1_inv@epsi_c1_tilde
# -

# ## Compute the time series of the cluster 3 aggregating variable

eig, _ = pcacov(rho2_c3)
e = eig[:, :k_c3].T
z_tilde_c3 = e@epsi_c3_tilde

# ## Compute the MLFP estimate of the correlation matrix of the aggregating variable

# +
z_c1 = zeros(z_tilde_c1.shape)
z_c3 = zeros(z_tilde_c3.shape)
for i in range(k_c1):
    z_c1[i, :] = t.ppf(t.cdf(z_tilde_c1[i, :], nu_c1), nu_aggr)

for i in range(k_c3):
    z_c3[i, :] = t.ppf(t.cdf(z_tilde_c3[i, :], nu_c3), nu_aggr)

_,sig2_aggr = DiffLengthMLFP(r_[z_c1, z_c3], p, nu_aggr, 10**-6)
rho2_aggr = np.diagflat(diag(sig2_aggr) ** (-1 / 2))@sig2_aggr@np.diagflat(diag(sig2_aggr) ** (-1 / 2))
