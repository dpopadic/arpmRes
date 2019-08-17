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

# # S_ProjectionHybridDefaultableZCB [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_ProjectionHybridDefaultableZCB&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-proj-hybrid-defaultable-zcb).

# ## Prepare the environment

# +
import os
import os.path as path
import sys
from collections import namedtuple

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import arange, ones, zeros, where, cumsum, diff, diag, eye, abs, round, log, exp, sqrt, tile, r_, array, \
    newaxis, histogram
from numpy import sum as npsum
from numpy.random import rand, randn

from tqdm import trange

from scipy.stats import norm, t as tstu, chi2
from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, xlim, ylim

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict
from FPmeancov import FPmeancov
from Price2AdjustedPrice import Price2AdjustedPrice
from FactorAnalysis import FactorAnalysis
from MaxLikelihoodFPLocDispT import MaxLikelihoodFPLocDispT
from CopMargSep import CopMargSep
from ProjectTransitionMatrix import ProjectTransitionMatrix
# -

# ## Run the script that computes and projects the market risk drivers

from S_ProjectionBootstrap import *

# ## Input

Bonds.ratings_tnow = [6,2,5,3,6]  # the ratings of the 5 ZCB are[B AA BB A B]

# ## Load the transition matrix estimated in S_FitDiscreteMarkovChain and "inject" it to a daily step
# ##(since projection step = 1 day, we work with a daily transition matrix)
# ##Load the transition matrix

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_FitCreditTransitions'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_FitCreditTransitions'), squeeze_me=True)

p_EP = db['p_EP']

try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_Ratings'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_Ratings'), squeeze_me=True)

db_Ratings = struct_to_dict(db['db_Ratings'])

Transitions = namedtuple('Transition', 'p ratings p_tau_step p_default')
Transitions.p = p_EP
Transitions.ratings = db_Ratings.ratings

# Inject the transition matrix
Transitions.p_tau_step = ProjectTransitionMatrix(Transitions.p, 1 / 252)
Transitions.p_default = Transitions.p_tau_step[:,-1]

# Compute threshold matrix
u_tau = r_['-1', zeros((Transitions.p_tau_step.shape[0], 1)), cumsum(Transitions.p_tau_step, 1)]
# -

# ## Compute the log-returns (invariants) of the 5 obligor's stocks

# +
n_issuers = len(Bonds.ratings_tnow)
n_dataset = StocksSPX['Prices'].shape[0]  # number of stocks in the dataset (we assume that the stocks of the 5 obligors are the last 5 entries in the dataset)
v_stocks_issuers = StocksSPX['Prices'][-n_issuers:,:]
# adjust prices for dividends
v_stocks_adj = zeros((n_issuers,t_+1))
for n in range(n_issuers):
    v_stocks_adj[n,:]=Price2AdjustedPrice(StocksSPX['Date'].reshape(1,-1), v_stocks_issuers[[n], :], StocksSPX['Dividends'][n_dataset - n_issuers + n])[0]

# we assume that log dividend-adjusted-values follow a random walk
epsi = diff(log(v_stocks_adj), 1, axis=1)  # invariants
bonds_i_ = epsi.shape[0]
# -

# ## Standardize the invariants

# +
nu_marg = 6
tauHL_prior = 252*3  # 3 years
# set FP
p_t = exp(-(log(2) / (tauHL_prior + round(10*(rand() - 0.5)))*abs(arange(t_, 1 + -1, -1)))).reshape(1,-1)  # FP setting for every invariants separately
p_t = p_t / npsum(p_t)

mu = zeros((bonds_i_, 1))
sig2 = zeros((bonds_i_, 1))
epsi_t = zeros((epsi.shape))
u = zeros((epsi.shape))
for i in range(bonds_i_):
    mu[i], sig2[i],_ = MaxLikelihoodFPLocDispT(epsi[[i],:], p_t, nu_marg, 10 ** -6, 1)
    epsi_t[i, :] = (epsi[i, :] - mu[i]) / sqrt(sig2[i])
    u[i, :] = tstu.cdf(epsi_t[i, :], nu_marg)
# -

# ## Estimate the correlation of the t-copula
# ## map observations into copula realizations

# +
nu = 5
c = zeros((u.shape))
for i in range(bonds_i_):
    c[i,:] = tstu.ppf(u[i, :], nu)

    # estimate the correlation matrix
[_, s2_hat] = FPmeancov(c, ones((1, t_)) / t_)
c2 = np.diagflat(1 / sqrt(diag(s2_hat)))@s2_hat@np.diagflat(1 / sqrt(diag(s2_hat)))
# -

# ## Factor analysis

# +
k_LRD = 1  # one factor
c2_LRD, beta,*_ = FactorAnalysis(c2, array([[0]]), k_LRD)
c2_LRD, beta = np.real(c2_LRD), np.real(beta)
c2_credit = np.diagflat(diag(c2_LRD) ** (-1 / 2))@c2_LRD@np.diagflat(diag(c2_LRD) ** (-1 / 2))
sig_credit = sqrt(diag(eye(c2_credit.shape[0]) - beta@beta.T))

Transitions.beta = beta
Transitions.c2_diag = diag(diag(eye((n_issuers)) - beta@beta.T))
Transitions.n_issuers = n_issuers
Transitions.n_ratings = Transitions.p.shape[0]
# -

# ## Copula marginal projection

# +
k_ = tau_proj  # 21 days
spx_idx = Stocks.i_ + Bonds.i_  # index of the S&P scenarios

T = zeros((bonds_i_, j_, k_))
for k in range(k_):
    # scenarios for the denominator
    M = chi2.ppf(rand(1, j_), nu)

    # scenarios for residuals.T numerator
    N_res = zeros((bonds_i_, j_))
    for i in range(bonds_i_):
        N_res[i,:] = randn(1, j_)

    # scenarios for the factor (S&P500 index already projected via Bootstrap)
    _, _, U_SPX = CopMargSep(Epsi_path[spx_idx, [k], :j_], p)  # standardize scenarios
    N_fac = norm.ppf(U_SPX, 0, 1)  # map scenarios into standard normal

    # compute joint scenarios
    T[:,:,k] = beta@(N_fac / tile(sqrt(M / nu)[newaxis,...], (k_LRD, 1))) + tile(sig_credit[...,newaxis], (1, j_)) * (N_res / tile(sqrt(M / nu)[newaxis,...],(n_issuers, 1)))

# map scenarios into grades
Epsi_credit = tstu.cdf(T, nu)
# -

# ## Translate scenarios into rating paths through the threshold matrix

Bonds.RatingProj = zeros((Transitions.n_issuers,j_,k_),dtype=int)
Bonds.I_D = zeros((Transitions.n_issuers,j_,k_))
for k in trange(k_,desc='Day'):
    for j in range(j_):
        for n in range(Transitions.n_issuers):
            if k == 0:
                Bonds.RatingProj[n, j, k]=where(histogram(Epsi_credit[n, j, k], u_tau[Bonds.ratings_tnow[n],:])[0] == 1)[0]
            else:
                Bonds.RatingProj[n, j, k] = where(histogram(Epsi_credit[n, j, k], u_tau[Bonds.RatingProj[n, j, k - 1],:])[0] == 1)[0]

    # Default indicator
    Bonds.I_D[:,:,k] = Bonds.RatingProj[:,:,k] == 8  # scenarios with rating=8 correspond to default

# ## Plot projected ratings

# +
gray = [.7, .7, .7]

figure()
plot(range(1,k_+1), Bonds.RatingProj[4, :,:].T)
plt.yticks(arange(10))
plt.xticks(arange(0, 25, 5))
xlim([0, k_ + 1])
ylim([0, 9]);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

