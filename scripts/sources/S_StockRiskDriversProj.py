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

# # S_StockRiskDriversProj [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_StockRiskDriversProj&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-garchdccinv-proj).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import arange, zeros, argsort, diff, diag, eye, abs, log, exp, sqrt, tile, array
from numpy import sum as npsum
from numpy.linalg import cholesky, pinv

from scipy.stats import t as tstu
from scipy.io import loadmat, savemat

import matplotlib.pyplot as plt

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import struct_to_dict
from intersect_matlab import intersect
from ConditionalFP import ConditionalFP
from MaxLikelihoodFPLocDispT import MaxLikelihoodFPLocDispT
from FactorAnalysis import FactorAnalysis
from Tscenarios import Tscenarios

# parameters

tauHL_smoo = 30  # half-life time for smoothing
tauHL_scor = 100  # half-life time for scoring

alpha = 0.25
tauHL_prior = 21*4  # parameters for Flexible Probabilities conditioned on VIX

nu_vec = arange(2,31)
nu_ = len(nu_vec)

nu_copula = 15  # degrees of freedom of t copula
k_ = 15  # factors for dimension reduction
m_ = 5  # monitoring times
j_ = 10  # number of scenarios
# -

# ## Upload databases

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_S&P500GARCHDCCInv'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_S&P500GARCHDCCInv'), squeeze_me=True)

dates = db['dates']
epsi_stocks = db['epsi_stocks']
q2_last= db['q2_last']
a_DCC = db['a_DCC']
b_DCC = db['b_DCC']
c_DCC = db['c_DCC']
sig2_GARCH = db['sig2_GARCH']
par_GARCH = db['par_GARCH']
deltax = db['deltax']

try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_VIX'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_VIX'), squeeze_me=True)

VIX = struct_to_dict(db['VIX'])
# -

# ## Compute the projected path scenarios via copula marginal/Monte Carlo approach

# +
# Flexible probabilities

# time series of conditioning variable (smoothed and scored VIX's
# compounded returns)
c_VIX = diff(log(VIX.value))
t_vix = len(c_VIX)
times = range(t_vix)

# smoothing
z_vix = zeros((1, t_vix))
for t in range(t_vix):
    p_smoo_t = exp(-log(2)/tauHL_smoo*(tile(t+1, (1, t+1))-times[:t+1]))
    gamma_t = npsum(p_smoo_t)
    z_vix[0,t] = npsum(p_smoo_t * c_VIX[:t+1]) / gamma_t

# scoring
mu_hat = zeros((1, t_vix))
mu2_hat = zeros((1, t_vix))
sd_hat = zeros((1, t_vix))
for t in range(t_vix):
    p_scor_t = exp(-log(2) / tauHL_scor*(tile(t+1, (1, t+1))-times[:t+1]))
    gamma_scor_t = npsum(p_scor_t)
    mu_hat[0,t] = npsum(p_scor_t * z_vix[0,:t+1]) / gamma_scor_t
    mu2_hat[0,t] = npsum(p_scor_t * (z_vix[0,:t+1]) ** 2) / gamma_scor_t
    sd_hat[0,t] = sqrt(mu2_hat[0,t]-(mu_hat[0,t]) ** 2)

z_vix = (z_vix - mu_hat) / sd_hat

# time series of invariants and VIX time series matching
dates_stocks, tau_vix, tau_stock = intersect(VIX.Date[1:], dates)
epsi_stocks = epsi_stocks[:, tau_stock]
z_vix = z_vix[[0],tau_vix]
z_vix_star = z_vix[-1]  # target value
i_, t_ = epsi_stocks.shape

# state and time conditioned probabilities
prior = exp(-log(2) / tauHL_prior*abs(arange(t_, 1 + -1, -1))).reshape(1,-1)
prior = prior / npsum(prior)

# conditioner
conditioner = namedtuple('conditioner', ['Series', 'TargetValue', 'Leeway'])
conditioner.Series = z_vix.reshape(1,-1)
conditioner.TargetValue = np.atleast_2d(z_vix_star)
conditioner.Leeway = alpha

p = ConditionalFP(conditioner, prior)

# marginal distribution fit
nu_marg = zeros(i_)
mu_marg = zeros(i_)
sig2_marg = zeros(i_)
for i in range(i_):
    mu_nu = zeros(nu_)
    sig2_nu = zeros(nu_)
    like_nu = zeros((1, nu_))
    for k in range(nu_):
        nu = nu_vec[k]
        mu_nu[k], sig2_nu[k],_ = MaxLikelihoodFPLocDispT(epsi_stocks[[i],:], p, nu, 10 ** -6, 1)
        epsi_t = (epsi_stocks[i,:]-mu_nu[k]) / sqrt(sig2_nu[k])
        like_nu[0,k] = npsum(p * log(tstu.cdf(epsi_t, nu) / sqrt(sig2_nu[k])))

    k_nu = argsort(like_nu[0])[::-1]
    nu_marg[i] = max(nu_vec[k_nu[0]], 10)
    mu_marg[i] = mu_nu[k_nu[0]]
    sig2_marg[i] = sig2_nu[k_nu[0]]

# Realized marginals mapping into standard Student t realizations
u_stocks = zeros((i_, t_))
epsi_tilde_stocks = zeros((i_, t_))
for i in range(i_):
    # u_stocks([i,:])=min((t.cdf((epsi_stocks[i,:]-mu_marg[i])/sqrt(sig2_marg[i]),nu_marg[i]),0.999))
    u_stocks[i,:]=tstu.cdf((epsi_stocks[i,:]-mu_marg[i]) / sqrt(sig2_marg[i]), nu_marg[i])
    epsi_tilde_stocks[i,:] = tstu.ppf(u_stocks[i,:], nu_copula)  # Student t realizations

# Correlation matrix characterizing the t copula estimation

# approximate the fit to normal in case of badly scaled warnings
_, sig2,_ = MaxLikelihoodFPLocDispT(epsi_tilde_stocks, p, 1e9, 1e-6, 1)
rho2 = np.diagflat(diag(sig2) ** (-1 / 2))@sig2@np.diagflat(diag(sig2) ** (-1 / 2))

# Shrink the correlation matrix towards a low-rank-diagonal structure
rho2, beta, *_ = FactorAnalysis(rho2, array([[0]]), k_)
rho2, beta = np.real(rho2), np.real(beta)

# Monte Carlo scenarios for each path node from the t copula
Epsi_tilde_hor = zeros((i_, m_, j_))
optionT = namedtuple('option', 'dim_red stoc_rep')
optionT.dim_red = 0
optionT.stoc_rep = 0
for m in range(m_):
    Epsi_tilde_hor[:,m,:]=Tscenarios(nu_copula, zeros((i_, 1)), rho2, j_, optionT)  # We simulate scenarios one node at a time

# Projected path scenarios
Epsi_stocks_hor = zeros((i_, m_, j_))
U_stocks_hor = zeros((i_, m_, j_))
for i in range(i_):
    for m in range(m_):
        U_stocks_hor[i, m,:]=tstu.cdf(Epsi_tilde_hor[i, m,:], nu_copula)
        Epsi_stocks_hor[i, m,:]=mu_marg[i] + sqrt(sig2_marg[i])*tstu.ppf(U_stocks_hor[i, m,:], nu_marg[i])
# -

# ## Retrieve the projected path scenarios for the quasi-invariants

# +
#inverse correlation matrix
delta2 = diag(eye(i_) - beta@beta.T)
omega2 = np.diagflat(1 / delta2)
rho2_inv = omega2 - omega2@beta.dot(pinv(beta.T@omega2@beta + eye(k_)))@beta.T@omega2

Xi = zeros((i_,m_,j_))
# quasi invariants
for j in range(j_):
    for m in range(m_):
        if m == 0:
            q2_prior=q2_last
            q2=c_DCC*rho2+b_DCC*q2_prior+a_DCC*epsi_stocks[:,-1]@epsi_stocks[:, -1].T
        else:
            q2 = c_DCC*rho2 + b_DCC*q2_prior + a_DCC*Epsi_stocks_hor[:, m, j]@Epsi_stocks_hor[:, m, j].T

        r2 = np.diagflat(diag(q2) ** (-1 / 2))@q2@np.diagflat(diag(q2) ** (-1 / 2))
        Xi[:, m, j]=cholesky(r2)@rho2_inv@Epsi_stocks_hor[:, m, j]
        q2_prior = q2
# -

# ## Compute the projected path scenarios of the risk drivers

X_hor = zeros((i_, m_, j_))
for i in range(i_):
    for j in range(j_):
        for m in range(m_):
            if m == 0:
                dX_hor_prior=deltax[i,-1]-deltax[i, -2]
                Sig2_prior=sig2_GARCH[i, -1]
                Sig2=par_GARCH[0, i]+par_GARCH[1, i]*Sig2_prior+par_GARCH[2, i]*dX_hor_prior**2
                X_hor[i, m, j]=sqrt(Sig2)*Xi[i, m, j]
            elif m == 1:
                dX_hor_prior = X_hor[i, m - 1, j] - deltax[i,-1]
                Sig2_prior = Sig2
                Sig2 = par_GARCH[0, i] + par_GARCH[1, i]*Sig2_prior + par_GARCH[2, i]*dX_hor_prior**2
                X_hor[i, m, j] = sqrt(Sig2)*Xi[i, m, j]
            else:
                dX_hor_prior = X_hor[i, m - 1, j] - X_hor[i, m - 2, j]
                Sig2_prior = Sig2
                Sig2 = par_GARCH[0, i] + par_GARCH[1, i]*Sig2_prior + par_GARCH[2, i]*dX_hor_prior**2
                X_hor[i, m, j] = sqrt(Sig2)*Xi[i, m, j]

# ## Store the results

varnames_to_save = ['Epsi_stocks_hor','X_hor','U_stocks_hor','nu_marg','mu_marg','sig2_marg','epsi_stocks','dates_stocks']
vars_to_save = {varname: var for varname, var in locals().items() if isinstance(var,(np.ndarray,np.float,np.int))}
vars_to_save = {varname: var for varname, var in vars_to_save.items() if varname in varnames_to_save}
savemat(os.path.join(TEMPORARY_DB, 'db_GARCHDCCMCProj'),vars_to_save)
