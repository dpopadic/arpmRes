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

# # S_NonRobustSampleMeanCovJackknife [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_NonRobustSampleMeanCovJackknife&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerJackknifeclip).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import zeros, sort, argsort, cov, mean, r_
from numpy import min as npmin, max as npmax
from numpy.linalg import norm as linalgnorm

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, xlim, ylim, ylabel, \
    xlabel

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict
from PlotTwoDimEllipsoid import PlotTwoDimEllipsoid
from Price2AdjustedPrice import Price2AdjustedPrice
from GarchResiduals import GarchResiduals
# -

# ## Upload the database

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_Stocks'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_Stocks'), squeeze_me=True)

StocksSPX = struct_to_dict(db['StocksSPX'])
# -

# ## Compute the dividend-adjusted returns of two stocks

# +
t_ = 100

_, x_1 = Price2AdjustedPrice(StocksSPX.Date.reshape(1,-1), StocksSPX.Prices[[25],:], StocksSPX.Dividends[25])  # Cisco Systems Inc returns
_, x_2 = Price2AdjustedPrice(StocksSPX.Date.reshape(1,-1), StocksSPX.Prices[[5],:], StocksSPX.Dividends[5])  # General Electric returns
date = StocksSPX.Date[1:]

x_1 = x_1[[0],-t_:]
x_2 = x_2[[0],-t_:]
date = date[-t_:]
# -

# ## Compute the invariants using GARCH(1,1) fit

# +
epsi = GarchResiduals(r_[x_1,x_2])

mu_hist = mean(epsi, 1,keepdims=True)
sigma2_hist = cov(epsi)
# -

# ## Compute the jackknife estimators

# +
epsi_jack = {}
mu_jack = {}
sigma2_jack = {}
norm_cov = zeros(t_)
for t in range(t_):
    epsi_jack[t] = np.delete(epsi,t,axis=1)
    mu_jack[t] = mean(epsi_jack[t], 1, keepdims=True)  # jackknife mean
    sigma2_jack[t] = cov(epsi_jack[t])  # jackknife covariance
    norm_cov[t] = linalgnorm(sigma2_hist - sigma2_jack[t], ord='fro')  # computation of the distance between the historical and the jackknife covariance estimators

# sort the covariance matrices so that the algorithm can select those
# which differ the most from the historical one
normsort, i_normsort = sort(norm_cov)[::-1], argsort(norm_cov)[::-1]
# -

# ## Generate figures comparing the historical ellipsoid defined by original data with the jackknife ellipsoid defined by perturbed data.

# +
k_ = 3  # number of figures

for k in range(k_):
    figure()
    e_jack = epsi_jack[i_normsort[k_+1-k]]

    # scatter plot with ellipsoid superimposed
    o_1 = plot(e_jack[0], e_jack[1], markersize=2.1,color=[0.4, 0.4, 0.4], marker='.',linestyle='none')
    o_2= plot(epsi[0, i_normsort[k_-(k+1)]], epsi[1, i_normsort[k_-(k+1)]], markersize= 10, color='r',marker='*',linestyle='none')
    xlim([1.1*npmin(epsi[0]), 1.1*npmax(epsi[0])])
    ylim([1.1*npmin(epsi[1]), 1.1*npmax(epsi[1])])
    xlabel('$\epsilon_1$')
    ylabel('$\epsilon_2$')
    ell_1=PlotTwoDimEllipsoid(mu_hist, sigma2_hist, 1, 0, 0, 'b', 1.5)  # historical ellipsoid
    ell_2=PlotTwoDimEllipsoid(mu_jack[i_normsort[k_-(k+1)]], sigma2_jack[i_normsort[k_-(k+1)]],1, 0, 0, 'r', 1.5)  # jackknife ellipsoid

    # leg
    leg=legend(['historical observations','removed observation','historical ellipsoid','jackknife ellipsoid']);
    # save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

