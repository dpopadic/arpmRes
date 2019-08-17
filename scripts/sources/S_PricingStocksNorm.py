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

# # S_PricingStocksNorm [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_PricingStocksNorm&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-pricing-stocks-normal).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import ones, diff, cov, round, mean, log, exp, tile
from numpy.random import multivariate_normal as mvnrnd

from scipy.io import loadmat, savemat

import matplotlib.pyplot as plt
from matplotlib.pyplot import bar, subplots, title

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict
from HistogramFP import HistogramFP

# parameters
n_ = 2  # number of selected stocks
indexes = [0, 1]  # indexes of selected stocks
tau = 20  # projection horizon
# -

# ## Upload the historical series of the daily dividend-adjusted stock values

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_StocksS_P'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_StocksS_P'), squeeze_me=True)

Data = struct_to_dict(db['Data'])
# -

# ## Select the observations corresponding to the first two stocks and compute the one-step invariants.
# ## Further, where the corresponding mean and covariance

x = log(Data.Prices[indexes,:])
dx = diff(x, 1, 1)
mu = mean(dx, 1)
sigma2 = cov(dx)

# ## Simulate j_=10000 Monte Carlo scenarios for the risk drivers.T scenarios at the horizon (20 days ahead)
# ## by using that the risk drivers at the horizon are normally distributed

j_ = 10000
x_tnow = log(Data.Prices[indexes, -1])
mu_tau = tau*mu
sigma2_tau = tau*sigma2
X_thor = tile(x_tnow[...,np.newaxis], (1, j_)) + mvnrnd(mu_tau, sigma2_tau, j_).T

# ## Compute the j_ Monte Carlo scenarios for the stocks' values at the horizon
# ## and the corresponding P&L's scenarios

v_tnow = Data.Prices[indexes, -1]
V_thor = exp(tile(log(v_tnow[...,np.newaxis]), (1, j_)) + X_thor - tile(x_tnow[...,np.newaxis], (1, j_)))
PL = V_thor - tile(v_tnow[...,np.newaxis], (1, j_))

# ## Save the data in db_StocksNormal

vars_to_save = {varname: var for varname, var in locals().items() if isinstance(var,(np.ndarray,np.float,np.int))}
savemat(os.path.join(TEMPORARY_DB,'db_StocksNormal'),vars_to_save)

# ## Plot the histograms of the stocks P&L's at the horizon.

# +
f, ax = subplots(2,1)

lgray = [.7, .7, .7]  # light gray
dgray = [.5, .5, .5]  # dark gray

# histogram of the first zero coupon bond P&L
plt.sca(ax[0])
n_bins = round(15*log(j_))  # number of histogram bins
option = namedtuple('option', 'n_bins')
option.n_bins = n_bins
[pdf1_mc, bin1_mc] = HistogramFP(PL[[0]], 1 / j_*ones((1, j_)), option)
bar(bin1_mc[:-1], pdf1_mc[0], width=bin1_mc[1]-bin1_mc[0],facecolor= lgray, edgecolor= dgray)
title('First stock: distribution of the P & L at the horizon = %.0f days' %tau)

# histogram of the second zero coupon bond P&L
plt.sca(ax[1])
n_bins = round(15*log(j_))  # number of histogram bins
option = namedtuple('option', 'n_bins')
option.n_bins = n_bins
[pdf2_mc, bin2_mc] = HistogramFP(PL[[1]], 1 / j_*ones((1, j_)), option)
bar(bin2_mc[:-1], pdf2_mc[0], width=bin2_mc[1]-bin2_mc[0],facecolor= lgray, edgecolor= dgray)
title('Second stock: distribution of the P & L at the horizon = %.0f days' %tau)
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
