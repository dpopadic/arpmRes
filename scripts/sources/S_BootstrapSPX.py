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

# # S_BootstrapSPX [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_BootstrapSPX&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-hist-boot-proj-vue).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import arange, zeros, argsort, diff, abs, log, exp, sqrt, tile, r_
from numpy import sum as npsum

from scipy.stats import t as tstu
from scipy.io import loadmat, savemat

import matplotlib.pyplot as plt

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from intersect_matlab import intersect
from ConditionalFP import ConditionalFP
from MaxLikelihoodFPLocDispT import MaxLikelihoodFPLocDispT
from SampleScenProbDistribution import SampleScenProbDistribution

def struct_to_dict(s, as_namedtuple=True):
    if as_namedtuple:
        if s.dtype.names:
            nt = namedtuple('db', s.dtype.names)
            d = {}
            for x in s.dtype.names:
                try:
                    if x in ['Parameters','marginalt','DCCfit']:
                        d[x] = struct_to_dict(s[x])
                    elif isinstance(s[x], np.ndarray):
                        if x == 'sig2':
                            d[x] = s[x][0]
                        else:
                            d[x] = s[x]
                    else:
                        d[x] = np.atleast_1d(s[x]).flatten()[0]
                except:
                    d[x] = None
            nt = nt(**d)
            return nt
    else:
        if s.dtype.names:
            return {x: np.atleast_1d(s[x]).flatten()[0] for x in s.dtype.names}

# parameters
tauHL_smoo = 30  # half-life time for smoothing
tauHL_scor = 100  # half-life time for scoring

alpha = 0.25
tauHL_prior = 21*4  # parameters for Flexible Probabilities conditioned on VIX

nu_vec = arange(2,31)
nu_ = len(nu_vec)

j_ = 100  # number of scenarios of projected invariants
m_ = 500  # number of monitoring times in the future
# -

# ## Upload database

# +
# this database is created by VUEscript
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_SPX_zcb_Invariants'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_SPX_zcb_Invariants'), squeeze_me=True)

dates = db['dates']
epsi_SPX = db['epsi_SPX']

try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_VIX'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_VIX'), squeeze_me=True)

from ARPM_utils import struct_to_dict

VIX = struct_to_dict(db['VIX'])
# -

# ## Recover the time series of realized invariants

# VIX's compounded returns
c_VIX = diff(log(VIX.value))

# ## Compute the time series of the conditioning variable by applying sequentially smoothing and scoring filters to the time series of VIX's compounded returns

# +
t_vix = len(c_VIX)
times = arange(t_vix)

# smoothing
z_vix = zeros((1, t_vix))
for it in range(t_vix):
    p_smoo_t = exp(-log(2) / tauHL_smoo * (tile(it + 1, (1, it + 1)) - times[:it + 1]))
    gamma_t = npsum(p_smoo_t)
    z_vix[0, it] = npsum(p_smoo_t * c_VIX[:it + 1]) / gamma_t

# scoring
mu_hat = zeros((1, t_vix))
mu2_hat = zeros((1, t_vix))
sd_hat = zeros((1, t_vix))
for t in range(t_vix):
    p_scor_t = exp(-log(2) / tauHL_scor*(tile(t+1, (1, t+1))-times[:t+1]))
    gamma_scor_t = npsum(p_scor_t)
    mu_hat[0,t] = npsum(p_scor_t * z_vix[0,:t+1]) / gamma_scor_t
    mu2_hat[0,t] = npsum(p_scor_t * (z_vix[0,:t+1])**2) / gamma_scor_t
    sd_hat[0,t] = sqrt(mu2_hat[0,t]-(mu_hat[0,t])**2)

z_vix = (z_vix - mu_hat) / sd_hat
dates_zvix=VIX.Date
# -

# ## Match the time series of invariants with the time series of the conditioning variable

dates_SPX, tau_vix, tau_SPX = intersect(VIX.Date, dates)
z_vix_cond=z_vix[[0],tau_vix].reshape(1,-1)
epsi_SPX=epsi_SPX[tau_SPX].reshape(1,-1)
i_, t_ = epsi_SPX.shape

# ## Compute the state and time conditioning probabilities

z_vix_star = z_vix_cond[[0],-1]  # target value
prior = exp((-(log(2) / tauHL_prior))*abs(arange(t_, 1 + -1, -1)))
prior = prior / npsum(prior)
# conditioner
conditioner = namedtuple('conditioner', 'Series TargetValue Leeway')
conditioner.Series = z_vix_cond
conditioner.TargetValue = np.atleast_2d(z_vix_star)
conditioner.Leeway = alpha
p = ConditionalFP(conditioner, prior)

# ## Estimate the marginal distributions

nu_marg_SPX = zeros(i_)
mu_marg_SPX = zeros(i_)
sig2_marg_SPX = zeros(i_)
for i in range(i_):
    mu_nu = zeros(nu_)
    sig2_nu = zeros(nu_)
    like_nu = zeros(nu_)
    for k in range(nu_):
        nu = nu_vec[k]
        mu_nu[k], sig2_nu[k],_ = MaxLikelihoodFPLocDispT(epsi_SPX[[i],:], p, nu, 10 ** -6, 1)
        epsi_t = (epsi_SPX[i, :] - mu_nu[k]) / sqrt(sig2_nu[k])
        like_nu[k] = npsum(p * log(tstu.pdf(epsi_t, nu) / sqrt(sig2_nu[k])))

    k_nu = argsort(like_nu)[::-1]
    nu_marg_SPX[i] = max(nu_vec[k_nu[0]], 10)
    mu_marg_SPX[i] = mu_nu[k_nu[0]]
    sig2_marg_SPX[i] = sig2_nu[k_nu[0]]

# ## Compute the historical distribution of the invariants' copula

u_SPX = zeros((i_, t_))
for i in range(i_):
    u_SPX[i,:]=tstu.cdf((epsi_SPX[i, :] - mu_marg_SPX[i]) / sqrt(sig2_marg_SPX[i]), nu_marg_SPX[i])

# ## Generate the grades' projected paths scenarios via historical bootstrapping

U_SPX_hor = zeros((i_, m_, j_))
for m in range(m_):
    U_boot= SampleScenProbDistribution(u_SPX, p, j_)
    U_SPX_hor[:,m,:] = U_boot.copy()

# ## Compute the projected path scenarios

Epsi_SPX_hor = zeros((i_, m_, j_))
for i in range(i_):
    for m in range(m_):
        Epsi_SPX_hor[i, m,:]=mu_marg_SPX[i] + sqrt(sig2_marg_SPX[i])*tstu.ppf(U_SPX_hor[i, m,:], nu_marg_SPX[i])

# ## Save the results

varnames_to_save = ['nu_marg_SPX', 'mu_marg_SPX', 'sig2_marg_SPX', 'U_SPX_hor', 'epsi_SPX_hor', 'epsi_SPX', 'dates_SPX', 'z_vix', 'dates_zvix']
vars_to_save = {varname: var for varname, var in locals().items() if isinstance(var,(np.ndarray,np.float,np.int))}
vars_to_save = {varname: var for varname, var in vars_to_save.items() if varname in varnames_to_save}
savemat(os.path.join(TEMPORARY_DB,'db_HistBootstrappingProj'),vars_to_save)
