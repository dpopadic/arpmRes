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

# # S_ProjectionMultivarGARCH [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_ProjectionMultivarGARCH&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-sim-mvgarch-proc).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

from numpy import array, ones, zeros, diff, eye, round, log, tile, r_
from numpy import min as npmin, max as npmax
from numpy.random import randn

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, bar, xlim, ylim, scatter, title, xticks, yticks, subplot

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import struct_to_dict, save_plot
from intersect_matlab import intersect
from HistogramFP import HistogramFP
from Price2AdjustedPrice import Price2AdjustedPrice
from Riccati import Riccati
from FitMultivariateGarch import FitMultivariateGarch
# -

# ## Load data

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_Stocks'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_Stocks'), squeeze_me=True)

SPX = struct_to_dict(db['SPX'])
StocksSPX = struct_to_dict(db['StocksSPX'])
# -

# ## Compute the log-returns

# +
SPX_ = SPX.Price_close  # S&P500
x1 = SPX_
dx1 = diff(log(x1))

x2, dx2 = Price2AdjustedPrice(StocksSPX.Date.reshape(1,-1), StocksSPX.Prices[[0]], StocksSPX.Dividends[0])  # Apple Inc

[date, i2, i3] = intersect(StocksSPX.Date[1:], SPX.Date[1:])
dx2 = dx2[[0],i2].reshape(1,-1)
dx1 = dx1[i3].reshape(1,-1)
# -

# ## Settings

j_ = 10000  # numbers of MC scenarios
n_ = 2  # numbers of securities
tau = 21  # projection horizon

# ## Estimate the daily compounded returns distribution

dx = r_[dx1, dx2]  # extract risk drivers increments (compounded returns)
demean = 1
eps = .01
df = 500
[m, a, b, c, sig2] = FitMultivariateGarch(dx, demean, eps, df)

# ## Project the compouded returns to a one-month horizon

# +
sig2_ = zeros((j_, n_, n_))
for j in range(j_):
    sig2_[j,:,:] = sig2.copy()

dx_j = zeros((n_, j_))
for t in range(tau):
    for j in range(j_):  # WARNING: this loop is for didactical purposes only. In real applications avoid looping
        #  compute new return
        epsi = randn(n_, 1)
        sig2 = sig2_[j,:,:]
        dx_temp = m + Riccati(eye(n_), sig2)@epsi
        dx_j[:, [j]] = dx_j[:, [j]] + dx_temp

        # update for next cycle
        s = (dx_temp - m)@(dx_temp - m).T
        sig2 = c + a * s + b * sig2
        sig2_[j,:,:] = sig2.copy()

X = tile(array([[x1[-1]], [x2[0,-1]]]), (1, j_)) + dx_j  # projected risk drivers
p = ones((1, j_)) / j_  # Flexible probabilities (flat)
# -

# ## Create figure

# +
colhist = [.9, .9, .9]
gray = [.5, .5, .5]

x_lim = [npmin(dx_j[0,:]), npmax(dx_j[0,:])]
y_lim = [npmin(dx_j[1,:]), npmax(dx_j[1,:])]

# Display results

figure()
# marginals
NumBins = round(10*log(j_))
# scatter plot
ax = plt.subplot2grid((3,3),(1,0), rowspan=2, colspan=2)
scatter(dx_j[0,:], dx_j[1,:], 3, c=gray, marker='*')
xlim(x_lim)
ylim(y_lim)
xticks([])
yticks([])
plt.grid(True)
title('Joint return distribution')
option = namedtuple('option', 'n_bins')
option.n_bins = NumBins
n2, d2 = HistogramFP(dx_j[[1],:], p, option)
ax = plt.subplot2grid((3,3),(1,2),rowspan=2)
plt.barh(d2[:-1], n2[0], height=d2[1]-d2[0], facecolor= colhist, edgecolor= 'k')
ylim(y_lim)
xticks([])
plt.text(1.1*npmax(n2), -0.02, 'Stock return distribution',rotation=-90)
ax = plt.subplot2grid((3,3),(0,0),colspan=2)
n1, d1 = HistogramFP(dx_j[[0],:], p, option)
bar(d1[:-1], n1[0], width=d1[1]-d1[0], facecolor= colhist, edgecolor= 'k')
xlim(x_lim)
yticks([]),
plt.text(-0.1, 1.1*npmax(n1), 'Market index return distribution')
plt.tight_layout(pad=3);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

