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

# # S_NonRobustSampleMeanCov [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_NonRobustSampleMeanCov&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerSampleMeanCovRob).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import array, zeros, cov, mean, r_

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
epsi = zeros((2,t_))

# epsi = GarchResiduals([x_1x_2])
epsi[0] = GarchResiduals(x_1)
epsi[1] = GarchResiduals(x_2, p0=[0, 0.1, 0.7])

mu_hist = mean(epsi, 1,keepdims=True)
sigma2_hist = cov(epsi)
# -

# ## Inclusion of additional observations within the dataset computation of perturbed sample mean and covariance

# +
y_1 = array([[max(epsi[0]),max(epsi[1])]]).T # first additional observation
y_2 = array([[max(epsi[0]),min(epsi[1])]]).T  # second additional observation
y_3 = array([[min(epsi[0]),min(epsi[1])]]).T  # third additional observation
y_4 = array([[min(epsi[0]),max(epsi[1])]]).T  # fourth additional observation
y = r_['-1',y_1, y_2, y_3, y_4]  # vector containing additional observations
k_ = y.shape[1]

epsi_y = zeros((2, t_ + 1, k_))  # dataset with additional observation
mu_y = zeros((2, k_))  # sample mean perturbed by additonal observation
sigma2_y = zeros((2, 2, k_))  # sample covariance perturbed by additional observation
for k in range(k_):
    epsi_y[:,:,k] = r_['-1',epsi, y[:,[k]]]
    mu_y[:,k] = mean(epsi_y[:,:,k], 1)
    sigma2_y[:,:,k] = cov(epsi_y[:,:,k])
# -

# ## Generate figures comparing the historical ellipsoid with the ellipsoid defined by perturbed data

for k in range(k_):
    figure()
    # scatter plot with ellipsoid superimposed
    o_1 = plot(epsi_y[0, :-1, k], epsi_y[1, : -1, k], markersize=5,color=[0.4, 0.4,0.4], marker='.',linestyle='none')

    o_2 = plot(epsi_y[0, -1, k], epsi_y[1, -1, k], markersize= 8, color='r',marker='.',linestyle='none')
    xlim([y_3[0] - 0.3, y_1[0] + 0.3])
    ylim([y_3[1] - 0.3, y_1[1] + 0.3])
    xlabel('$\epsilon_1$')
    ylabel('$\epsilon_2$')
    ell_1 = PlotTwoDimEllipsoid(mu_hist, sigma2_hist, 1, 0, 0, 'b', 1.5)  # historical ellipsoid
    ell_2 = PlotTwoDimEllipsoid(mu_y[:,[k]], sigma2_y[:,:,k], 1, 0, 0, 'r', 1.5)  # perturbed ellipsoid

    # leg
    leg = legend(handles=[o_1[0],ell_1[0][0],o_2[0],ell_2[0][0]],labels=['historical observations','historical ellipsoid','additional observation','ellipsoid with additional observation']);
    # save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

