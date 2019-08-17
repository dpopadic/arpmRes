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

# # S_FullyExogenousLFMBonds [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_FullyExogenousLFMBonds&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-full-exogen-lfm).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import reshape, ones, zeros, tril, diag, round, log, sqrt, r_, diff

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, bar, title

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot
from FPmeancov import FPmeancov
from HistogramFP import HistogramFP
# -

# ## Load data

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_BondAttribution'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_BondAttribution'), squeeze_me=True)

beta = db['beta']
dates = db['dates']
X_shift = db['X_shift']
Z = db['Z']
# -

# ## Compute residuals

# +
[n_, k_, t_] = beta.shape

U = zeros((n_, t_))
for t in range(t_):
    U[:,t] = X_shift[:,t] - beta[:,:, t]@Z[:, t]
# -

# ## Residuals analysis

# ## compute statistics of the joint distribution of residuals and factors
m_UZ, s2_UZ = FPmeancov(r_[U,Z], ones((1, t_)) / t_)

# ## compute correlation matrix

# +
sigma = sqrt(diag(s2_UZ))
c2_UZ = np.diagflat(1 / sigma)@s2_UZ@np.diagflat(1 / sigma)

c_UZ = c2_UZ[:n_, n_ :n_ + k_]
c2_U = tril(c2_UZ[:n_, :n_], -1)
# -

# ## Plot (untruncated) correlations among residuals

# +
# reshape the correlations in a column vector
corr_U = []
for i in range(1,n_):
    corr_U = r_[corr_U, c2_U[i:,i-1]]  # reshape the correlations in a column vector

nbins = round(5*log(len(corr_U)))
p = ones((1, len(corr_U))) / len(corr_U)
option = namedtuple('option', 'n_bins')

option.n_bins = nbins
n, xout = HistogramFP(corr_U[np.newaxis,...], p, option)

figure()
h = bar(xout[:-1]+diff(xout,1), n[0], width=xout[1]-xout[0],facecolor=[.7, .7, .7], edgecolor='k')
title('Correlations among residuals');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
# -

# ## Plot (untruncated) correlations between factors and residuals

# +
corr_UZ = reshape(c_UZ, (n_*k_, 1),'F')  # ## reshape the correlations in a column vector
nbins = round(5*log(len(corr_UZ)))
p = ones((1, len(corr_UZ))) / len(corr_UZ)
option = namedtuple('option', 'n_bins')
option.n_bins = nbins
n, xout = HistogramFP(corr_UZ.T, p, option)

figure()
h = bar(xout[:-1], n[0], width=xout[1]-xout[0],facecolor= [.7, .7, .7], edgecolor='k')
title('Correlations between factors residuals');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
