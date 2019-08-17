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

# # S_CMBootJoin [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_CMBootJoin&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-hyb-mchist-proj-vue).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

from tqdm import trange

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import arange, zeros, argsort, squeeze, \
    diag, eye, abs, log, exp, sqrt, newaxis, r_, array
from numpy import sum as npsum
from numpy.linalg import solve, pinv

from scipy.stats import t
from scipy.io import loadmat

import matplotlib.pyplot as plt

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from intersect_matlab import intersect
from FactorAnalysis import FactorAnalysis
from MaxLikelihoodFPLocDispT import MaxLikelihoodFPLocDispT
from ConditionalFP import ConditionalFP
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=S_CMBootJoin-parameters)

# +
nu_joint = 5
tauHL_smoo = 30  # half-life time for smoothing
tauHL_scor = 100  # half-life time for scoring

alpha = 0.25
tauHL_prior = 21*4  # parameters for Flexible Probabilities conditioned on VIX

nu_vec = range(2,31)
nu_ = len(nu_vec)

k_ = 15  # number of factors for factor analysis
# -

# ## Upload databases

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_GARCHDCCMCProj'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_GARCHDCCMCProj'), squeeze_me=True)

dates_stocks = db['dates_stocks']
epsi_stocks = db['epsi_stocks']
U_stocks_hor = db['U_stocks_hor']
nu_marg = db['nu_marg']
mu_marg = db['mu_marg']
sig2_marg = db['sig2_marg']

try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_HistBootstrappingProj'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_HistBootstrappingProj'), squeeze_me=True)

epsi_SPX = db['epsi_SPX'].reshape(1,-1)
dates_zvix = db['dates_zvix']
dates_SPX = db['dates_SPX']
z_vix = db['z_vix']
nu_marg_SPX = db['nu_marg_SPX']
mu_marg_SPX = db['mu_marg_SPX']
sig2_marg_SPX = db['sig2_marg_SPX']
if db['U_SPX_hor'].ndim == 2:
    U_SPX_hor = db['U_SPX_hor'][newaxis,...]
else:
    U_SPX_hor = db['U_SPX_hor']
# ## Intersect the times series of the one-step invariants and of the conditioning variable

[dates_epsi, tau_stocks, tau_SPX] = intersect(dates_stocks, dates_SPX)
epsi_stocks = epsi_stocks[:, tau_stocks]
epsi_SPX = epsi_SPX[:, tau_SPX]
epsi = r_[epsi_SPX, epsi_stocks]
_, tau_vix, tau_epsi = intersect(dates_zvix, dates_epsi)
z_vix_cond = z_vix[tau_vix]
epsi = epsi[:, tau_epsi]

i_, t_ = epsi.shape
i_stocks, _ = epsi_stocks.shape
i_SPX, _ = epsi_SPX.reshape(1,-1).shape
_, m_, j_ = U_stocks_hor.shape
# -

# ## Estimate the joint correlation matrix

# +
# flexible probabilities
z_vix_star = z_vix_cond[-1]  # target value
prior = exp(-log(2) / tauHL_prior*abs(arange(t_, 1 + -1, -1)))
prior = prior / npsum(prior)
# conditioner
conditioner = namedtuple('conditioner', ['Series', 'TargetValue', 'Leeway'])
conditioner.Series = z_vix_cond.reshape(1,-1)
conditioner.TargetValue = np.atleast_2d(z_vix_star)
conditioner.Leeway = alpha
p = ConditionalFP(conditioner, prior)

# map invariants into student t realizations
nu_marg = r_[nu_marg, nu_marg_SPX]
mu_marg = r_[mu_marg, mu_marg_SPX]
sig2_marg = r_[sig2_marg, sig2_marg_SPX]
epsi_tilde = zeros((i_,t_))
for i in range(i_):
    u=t.cdf((epsi[i,:]-mu_marg[i]) / sqrt(sig2_marg[i]), nu_marg[i])
    epsi_tilde[i,:]=t.ppf(u, nu_joint)

# estimate joint correlation
_, sig2,_ = MaxLikelihoodFPLocDispT(epsi_tilde, p, nu_joint, 10 ** -6, 1)
c = np.diagflat(diag(sig2) ** (-1 / 2))@sig2@np.diagflat(diag(sig2) ** (-1 / 2))

# replace the correlation block related to stocks with its low-rank-diagonal
# approximation
c_stocks, beta_stocks,*_ = FactorAnalysis(c[i_SPX:i_SPX + i_stocks, i_SPX:i_SPX+ i_stocks], array([[0]]), k_)
c_stocks, beta_stocks = np.real(c_stocks),np.real(beta_stocks)
c_SPX_stocks = c[:i_SPX, i_SPX :i_SPX + i_stocks]
c_SPX = c[:i_SPX, :i_SPX]
# -

# ## Perform Hybrid Monte-Carlo historical projection on the grades for each node path

Epsistocks_tilde_hor = zeros((i_stocks, U_stocks_hor.shape[2]))
EpsiSPX_tilde_hor = zeros((i_SPX, U_SPX_hor.shape[2]))
Ujoint_hor = zeros((i_,m_,j_))
for m in trange(m_):
    Ujoint_hor_node=zeros((i_, j_))
    # map projected grades into standard Student t realizations
    for i in range(i_stocks):
        Epsistocks_tilde_hor[i,:]=squeeze(t.ppf(U_stocks_hor[i, m,:], nu_joint))

    for i in range(i_SPX):
        EpsiSPX_tilde_hor[i,:]=squeeze(t.ppf(U_SPX_hor[i, m, :], nu_joint))

    # conditional historical expectation
    m_SPX = zeros((i_SPX, j_))

    # inverse stocks's correlation matrix from binomial theorem
    delta2 = diag(eye(i_stocks) - beta_stocks@beta_stocks.T)
    omega2 = np.diagflat(1 / delta2)
    c_stocks_inv = omega2 - omega2@beta_stocks.dot(pinv(beta_stocks.T@omega2@beta_stocks + eye(k_)))@beta_stocks.T@omega2

    m_SPX = c_SPX_stocks@c_stocks_inv@Epsistocks_tilde_hor

    # Squared Mahalanobis distances
    d = zeros(j_)
    for j in range(j_):
        d[j]=Epsistocks_tilde_hor[:,j].T@c_stocks_inv@Epsistocks_tilde_hor[:, j]

    j_sorted_stocks = argsort(d)[::-1]  # sort indexes accordind to d

    J = arange(j_)

    for j in range(j_):
        # index of the SPX's scenarios having greatest
        # observation wrt the corresponding conditional expectation
        d2=zeros((1, len(J)))
        for jj in range(len(J)):
            d2[0,jj]=solve(np.atleast_2d(EpsiSPX_tilde_hor[:,J[jj]]-m_SPX[:, j_sorted_stocks[j]]).T,c_SPX)@(EpsiSPX_tilde_hor[:, J[jj]]-m_SPX[:, j_sorted_stocks[j]])

        perm_j = argsort(d2[0])[::-1]
        j_SPX = J[perm_j[0]]

        # joint projected scenarios for the invariants' grades
        for i in range(i_SPX):
            Ujoint_hor_node[i, j_sorted_stocks[j]]=t.cdf(EpsiSPX_tilde_hor[i, j_SPX], nu_joint)

        for i in range(i_stocks):
            Ujoint_hor_node[i+i_SPX, j_sorted_stocks[j]]=t.cdf(Epsistocks_tilde_hor[i, j_sorted_stocks[j]], nu_joint)

        # discard index perm_j
        np.delete(J, perm_j[0])
    Ujoint_hor[:,m,:]=Ujoint_hor_node

# ## Compute the projected joint paths scenarios

Epsi_hor = zeros((i_,m_, Ujoint_hor.shape[2]))
for m in range(m_):
    for i in range(i_):
        Epsi_hor[i, m,:]=mu_marg[i] + sqrt(sig2_marg[i])*t.ppf(Ujoint_hor[i, m,:], nu_marg[i])
