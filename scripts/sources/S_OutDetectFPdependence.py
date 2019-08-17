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

# # S_OutDetectFPdependence [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_OutDetectFPdependence&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerMVEOutlier).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import arange, array, zeros, ceil, log, exp, tile, r_, linspace
from numpy import sum as npsum

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, bar, legend, xlim, ylim, scatter, ylabel, \
    xlabel, xticks, yticks
import matplotlib.dates as mdates

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict, date_mtop
from FPmeancov import FPmeancov
from PlotTwoDimEllipsoid import PlotTwoDimEllipsoid
from Price2AdjustedPrice import Price2AdjustedPrice
from GarchResiduals import GarchResiduals
from BlowSpinFP import BlowSpinFP
from ColorCodedFP import ColorCodedFP
from RemoveFarthestOutlierFP import RemoveFarthestOutlierFP
from FarthestOutlier import FarthestOutlier
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
n_ = 2
t_ = 500

_, x_1 = Price2AdjustedPrice(StocksSPX.Date.reshape(1,-1), StocksSPX.Prices[[25],:], StocksSPX.Dividends[25])  # Cisco Systems Inc returns
_, x_2 = Price2AdjustedPrice(StocksSPX.Date.reshape(1,-1), StocksSPX.Prices[[5],:], StocksSPX.Dividends[5])  # General Electric returns
date = StocksSPX.Date[1:]

x_1 = x_1[[0],-t_:]
x_2 = x_2[[0],-t_:]
date = date[-t_:]
# -

# ## Compute the invariants using GARCH(1,1) fit

epsi = GarchResiduals(r_[x_1,x_2])

# ## Compute the Flexible Probability profiles using Blow-Spin method

b = 1  # number of blows
s = 2  # number of spins
p, _ = BlowSpinFP(epsi, b, s, [1, 1], .8)
q_ = b + s

# ## Remove the worst historical outliers from the dataset to guarantee clarity in static figures

# +
for k in range(int(ceil(t_ / 15))):
    epsi, p, date = RemoveFarthestOutlierFP(epsi, p, date)

p=p / tile(npsum(p, 1,keepdims=True), (1, p.shape[1]))  # normalize the FP profiles
ens = exp(npsum(-p * log(p), 1,keepdims=True))  # compute the effective number of scenarios
# -

# ## Detect the worst outlier for each FP profile then compute HFP mean and covariance

t_tilde = zeros(q_,dtype=int)
mu_out = zeros((n_, q_))
sigma2_out = zeros((n_, n_, q_))
for q in range(q_):
    t_tilde[q] = FarthestOutlier(epsi, p[[q],:])  # where the time subscript of the worst outlier
    # compute historical mean and covariance of the dataset without outlier
    epsi_temp = np.delete(epsi,t_tilde[q], axis=1)
    p_temp = np.delete(p[[q],:],t_tilde[q], axis=1)
    [mu_out[:, [q]], sigma2_out[:,:, q]] = FPmeancov(epsi_temp, p_temp / npsum(p_temp))

# ## Generate static figures showing how the detected outlier changes along with the FP profile considered

# +
greyrange = arange(0.1,0.91,0.01)
date_dt = array([date_mtop(i) for i in date])
myFmt = mdates.DateFormatter('%d-%b-%Y')

t_new = len(date_dt)
epslim1 = [min(epsi[0]) - .3, max(epsi[0])+.3]
epslim2 = [min(epsi[1]) - .3, max(epsi[1])+.3]

for q in range(q_):
    f = figure()

    # Scatter plot of observations, outlier and HFP-ellipsoid
    plt.subplot2grid((4,1),(0,0),rowspan=3)
    [CM, C] = ColorCodedFP(p[[q],:], None, None, greyrange, 0, 1, [0.6, 0.1])
    # colormap(CM)
    obs = scatter(epsi[0], epsi[1], 8, c=C, marker='.',cmap=CM)

    shobs = plot(-1000, 1000, color='k',marker='.',markersize=8,linestyle='none')
    xlim(epslim1)
    ylim(epslim2)
    out = scatter(epsi[0, t_tilde[q]], epsi[1, t_tilde[q]], 50, 'r','o',lw=2)
    shout = plot(-1000, 1000, markersize= 6, color='r',marker='o',lw=2,linestyle='none')
    ell = PlotTwoDimEllipsoid(mu_out[:, [q]], sigma2_out[:,:, q], 1, None, None, 'r', 2)
    xlabel('$\epsilon_1$')
    ylabel('$\epsilon_2$')
    plt.grid(True)
    leg = legend(['historical observations','worst outlier','HFP ellipsoid'])

    # Flexible Probability profile
    plt.subplot(4,1,4)
    b = bar(date_dt, p[q, :],width=date_dt[1].toordinal()-date_dt[0].toordinal(), facecolor=[.6, .6, .6], edgecolor=[.6, .6, .6])
    d = linspace(0,t_new-1,3,dtype=int)
    xlim([min(date_dt), max(date_dt)])
    xticks(date_dt[d])
    plt.gca().xaxis.set_major_formatter(myFmt)
    ylim([0, max(p[q,:])])
    yticks([])
    ylabel('FP')
    ensT = 'Effective Num.Scenarios =  % 3.0f'%ens[q]
    plt.tight_layout();
    plt.text(date_dt[-1], max(p[q,:])+max(p[q, :]) / 10, ensT, color = 'k',horizontalalignment='right',verticalalignment='bottom')
    # save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
