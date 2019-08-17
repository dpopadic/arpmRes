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

# # S_GenCrossSectionEstimLFM [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_GenCrossSectionEstimLFM&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-3-ex-unc-cross-sec).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import reshape, ones, zeros, tril, diag, eye, round, log, tile, r_

from scipy.linalg import kron
from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, bar, title

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot
from FPmeancov import FPmeancov
from HistogramFP import HistogramFP
from quadprog import quadprog

# input parameters
n_ = 100  # target dimension
k_ = 10  # number of factors
i_n = eye(n_)
i_k = eye(k_)
# -

# ## Load weekly observations of the stocks

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_Securities_TS'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_Securities_TS'), squeeze_me=True)

data = db['data']
data_securities = data[1:,:]  # 1st row is date
# -

# ## Load sector-securities binary exposures

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_Securities_IndustryClassification'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_Securities_IndustryClassification'), squeeze_me=True)

data = db['data']
securities_industry_classification = data
# -

# ## Compute linear returns of stocks

V = data_securities[:n_,:]  # values
R = (V[:, 1:] - V[:, : -1]) / V[:, : -1]
_, t_ = R.shape
p = ones((1, t_)) / t_  # Flexible Probabilities

# ## Set the exogenous loadings

beta = securities_industry_classification[:n_,:k_]

# ## Solve quadratic programming problem

# +
km = zeros((k_*n_, k_*n_))  # commutation matrix
for n in range(n_):
    for k in range(k_):
        km = km + kron(i_k[:,[k]]@i_n[:, [n]].T, i_n[:,[n]] @i_k[:, [k]].T)  # set inputs for quadratic programming problem

[m_R, s2_R] = FPmeancov(R, p)
invsigma2 = np.diagflat(1 / diag(s2_R))
pos = beta.T@invsigma2@s2_R
g = -pos.flatten('F')
q = kron(s2_R, beta.T@invsigma2@beta)
q_, _ = q.shape

# linear constraints
v = ones((1, n_)) / n_
d_eq = kron(i_k, v@s2_R)@km
b_eq = zeros((k_, 1))

# compute extraction matrix
c = quadprog(q, g, d_eq, b_eq)

gamma = reshape(c, (k_, n_),'F')
Z = gamma@R
# -

# ## Compute shift parameter

# +
[mu_Z, sig2_Z] = FPmeancov(Z, p)

alpha = m_R - beta@mu_Z
# -

# ## Compute residuals

U = R - tile(alpha, (1, t_)) - beta@Z
[mu_UZ, sig2_UZ] = FPmeancov(r_[U, Z], p)  # sample joint covariance

# ## Compute correlations between factors and residuals, and correlations among residuals

# +
c2_UZ = np.diagflat(diag(sig2_UZ) ** (-1 / 2))@sig2_UZ@np.diagflat(diag(sig2_UZ) ** (-1 / 2))

c_UZ = c2_UZ[:n_, n_ :n_ + k_]
c2_U = tril(c2_UZ[:n_, :n_], -1)
# -

# ## Compute truncated covariance of returns

sig2_U = sig2_UZ[:n_, :n_]
sig2_Rtrunc = beta@sig2_Z@beta.T + np.diagflat(diag(sig2_U))

# ## Plot (untruncated) correlations among residuals
# ## reshape the correlations in a column vector

# +
corr_U = []
for i in range(1,n_):
    corr_U = r_[corr_U, c2_U[i:,i-1]]  # ## reshape the correlations in a column vector
corr_U = corr_U.reshape(-1,1)

nbins = round(5*log(len(corr_U)))
p = ones((1, len(corr_U))) / len(corr_U)
option = namedtuple('option', 'n_bins')
option.n_bins = nbins
[n, xout] = HistogramFP(corr_U.T, p, option)

figure()

h = bar(xout[:-1], n[0], width=xout[1]-xout[0],facecolor= [.7, .7, .7], edgecolor='k')
title('Correlation among residuals')
# -

# ## Plot (untruncated) correlations between factors and residuals

# +
# ## reshape the correlations in a column vector

corr_UZ = reshape(c_UZ, (n_*k_, 1),'F')
p = ones((1, len(corr_UZ))) / len(corr_UZ)
[n, xout] = HistogramFP(corr_UZ.T, p, option)

figure()
h = bar(xout[:-1], n[0], width=xout[1]-xout[0], facecolor= [.7, .7, .7], edgecolor='k')
title('Correlation factors-residuals');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
