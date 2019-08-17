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

# # S_GenRegEstimLFM [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_GenRegEstimLFM&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-3-ex-un-ts-ind-factor).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import reshape, ones, zeros, tril, diag, eye, round, log, sqrt, r_, array

from scipy.linalg import kron
from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, bar, title

from quadprog import quadprog
plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict
from FPmeancov import FPmeancov
from HistogramFP import HistogramFP

# input parameters
n_ = 100  # target dimension
k_ = 10  # number of factors
I_n = eye(n_)
I_k = eye(k_)
# -

# ## Load weekly observations of the stocks

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_Securities_TS'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_Securities_TS'), squeeze_me=True)

data = db['data']

data_securities = data[1:,:]  # 1st row is for date
# -

# ## Load weekly observations of sector indices

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_Sectors_TS'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_Sectors_TS'), squeeze_me=True)

data = db['data']

data_sectors = data[2:,:]  # 1st row is for date, 2nd row is SPX index
# -

# ## Compute linear returns of stocks

Vstock = data_securities[:n_,:]  # values
X = (Vstock[:, 1:] - Vstock[:, : -1]) / Vstock[:, : -1]
[_, t_] = X.shape
p = ones((1, t_)) / t_  # Flexible Probabilities

# ## Compute linear returns of sector indices

Vsector = data_sectors[:k_,:]  # values
Z = (Vsector[:, 1:] - Vsector[:, : -1]) / Vsector[:, : -1]

# ## Compute statistics of the joint distribution of X,Z

[m_XZ, s2_XZ] = FPmeancov(r_[X, Z], p)
s2_X = s2_XZ[:n_, :n_]
s_XZ = s2_XZ[:n_, n_:n_ + k_]
s2_Z = s2_XZ[n_ :n_ + k_, n_ :n_ + k_]

# ## Solve generalized regression LFM
# ## set inputs for quadratic programming problem

# +
d = np.diagflat(1 / diag(s2_X))
pos = d@s_XZ
g = -pos.flatten()
q = kron(s2_Z, d)
q_, _ = q.shape

# set constraints
a_eq = ones((1, n_*k_)) / (n_*k_)
b_eq = array([[1]])
lb = 0.8*ones((n_*k_, 1))
ub = 1.2*ones((n_*k_, 1))

# compute optimal loadings
b = quadprog(q, g, a_eq, b_eq, lb, ub)
b = np.array(b)

beta = reshape(b, (n_, k_),'F')
alpha = m_XZ[:n_] - beta@m_XZ[n_ :n_ + k_]
# -

# ## Residuals analysis
# ## compute statistics of the joint distribution of residuals and factors

# +
m = r_[r_['-1',I_n, - beta], r_['-1',zeros((k_, n_)), I_k]]
m_UZ = m@m_XZ - r_[alpha, zeros((k_, 1))]  # joint expectation
s2_UZ = m@s2_XZ@m.T  # joint covariance

# compute correlation matrix
sigma = sqrt(diag(s2_UZ))
c2_UZ = np.diagflat(1 / sigma)@s2_UZ@np.diagflat(1 / sigma)

c_UZ = c2_UZ[:n_, n_ :n_ + k_]
c2_U = tril(c2_UZ[:n_, :n_], -1)
# -

# ## Plot (untruncated) correlations among residuals
# ## reshape the correlations in a column vector

# +
corr_U = []
for i in range(1,n_):
    corr_U = r_[corr_U, c2_U[i:,i-1]]  # ## reshape the correlations in a column vector

nbins = round(5*log(len(corr_U)))
p = ones((1, len(corr_U))) / len(corr_U)
option = namedtuple('option', 'n_bins')
option.n_bins = nbins
[n, xout] = HistogramFP(corr_U.reshape(1,-1), p, option)

figure()

h = bar(xout[:-1], n[0], width=xout[1]-xout[0],facecolor= [.7, .7, .7],edgecolor='k')
title('Correlations among residuals')
# -

# ## Plot (untruncated) correlations between factors and residuals

corr_UZ = reshape(c_UZ, (n_*k_, 1),'F')

# ## reshape the correlations in a column vector

# +
p = ones((1, corr_UZ.shape[0])) / corr_UZ.shape[0]
[n, xout] = HistogramFP(corr_UZ.T, p, option)

figure()
h = bar(xout[:-1], n[0], width=xout[1]-xout[0],facecolor= [.7, .7, .7],edgecolor='k')
title('Correlations between factors residuals');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

