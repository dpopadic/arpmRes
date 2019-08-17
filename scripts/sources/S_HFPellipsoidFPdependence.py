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

# # S_HFPellipsoidFPdependence [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_HFPellipsoidFPdependence&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerHFPellipsoidPlot).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import arange, array, zeros, percentile, cov, round, mean, log, r_
from numpy import max as npmax
from numpy.linalg import solve

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, bar, xlim, ylim, scatter, subplots, ylabel, \
    xlabel, xticks, yticks
import matplotlib.dates as mdates

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import struct_to_dict, date_mtop, save_plot, matlab_percentile
from FPmeancov import FPmeancov
from PlotTwoDimEllipsoid import PlotTwoDimEllipsoid
from HistogramFP import HistogramFP
from Price2AdjustedPrice import Price2AdjustedPrice
from GarchResiduals import GarchResiduals
from ColorCodedFP import ColorCodedFP
from BlowSpinFP import BlowSpinFP
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
t_ = 300

_, x_1 = Price2AdjustedPrice(StocksSPX.Date.reshape(1,-1), StocksSPX.Prices[[25],:], StocksSPX.Dividends[25])  # Cisco Systems Inc
_, x_2 = Price2AdjustedPrice(StocksSPX.Date.reshape(1,-1), StocksSPX.Prices[[5],:], StocksSPX.Dividends[5])  # General Electric
date = StocksSPX.Date[1:]

x_1 = x_1[[0],-t_:]
x_2 = x_2[[0],-t_:]
date = date[-t_:]
# -

# ## Compute the invariants using GARCH(1,1) fit

# +
epsi = GarchResiduals(r_[x_1,x_2])

mu_hist = mean(epsi, 1)
sigma2_hist = cov(epsi.T)
# -

# ## Compute the Flexible Probability profiles using Blow-Spin method

b = 2  # number of blows
s = 3  # number of spins
p, ens = BlowSpinFP(epsi, b, s, [.5, 2], 0.8)

# ## Compute HFP-ellipsoid and HFP-histogram

# +
q_ = b + s
mu_HFP = zeros((2, q_))  # array containing the mean vector for each one of the q_ profiles
sigma2_HFP = zeros((2, 2, q_))  # array containing the covariance matrix for each one of the q_ profiles
z_2 = zeros((q_, t_))
mu_z2 = zeros((1, q_))

for q in range(q_):
    mu_HFP[:, [q]], sigma2_HFP[: ,:, q] = FPmeancov(epsi, p[[q],:])
    for t in range(t_):
        z_2[q, t] = (epsi[:,t]-mu_HFP[:, q]).T@solve(n_*sigma2_HFP[:,:, q],epsi[:,t]-mu_HFP[:, q])
    mu_z2[0,q] = p[q,:]@z_2[q,:].T
# -

# ## Generate some figures showing how the HFP-ellipsoid evolves as the FP profile changes

# +
grey_range = arange(0,0.81,0.01)
q_range = array([1, 99])
date_dt = array([date_mtop(i) for i in date])
myFmt = mdates.DateFormatter('%d-%b-%Y')

for q in range(q_):
    f, ax = subplots(2,2)
    P = p[q,:]
    # scatter colormap and colors
    CM, C = ColorCodedFP(P[np.newaxis,...], None, None, grey_range, 0, 1, [0.7, 0.3])

    # scatter plot of (epsi1,epsi2) with HFP-ellipsoid superimposed
    plt.subplot(121)
    # colormap(CM)
    plt.axis('equal')
    scatter(epsi[0], epsi[1], 15, c=C, marker='.',cmap=CM)
    xlim(percentile(epsi[0], q_range))
    ylim(percentile(epsi[1], q_range))
    xlabel('$\epsilon_1$')
    ylabel('$\epsilon_2$')
    PlotTwoDimEllipsoid(mu_HFP[:, [q]], sigma2_HFP[:,:, q], 1, 0, 0, 'r', 2)

    # histogram of z^2
    options = namedtuple('option', 'n_bins')
    options.n_bins = round(30*log(ens[0,q]))
    plt.sca(ax[0, 1])
    ax[0,1].set_facecolor('white')
    nz, zz = HistogramFP(z_2[[q], :], P.reshape(1,-1), options)
    b = bar(zz[:-1], nz[0],width=zz[1]-zz[0],facecolor=[.7, .7, .7], edgecolor=[.3, .3, .3])
    plt.axis([-1, 15, 0, npmax(nz) + (npmax(nz) / 20)])
    yticks([])
    xlabel('$z^2$')

    plot(mu_z2[0,q], 0, color='r',marker='o',markerfacecolor='r', markersize = 4)
    MZ2 = 'HFP - mean($z^2$) =  % 3.2f'%mu_z2[0,q]
    plt.text(15, npmax(nz) - (npmax(nz) / 7), MZ2, color='r',horizontalalignment='right',verticalalignment='bottom')

    # flexible probabilities profiles
    plt.sca(ax[1,1])
    ax[1,1].set_facecolor('white')
    b = bar(date_dt,P,width=date_dt[1].toordinal()-date_dt[0].toordinal(),facecolor=[.7, .7, .7], edgecolor=[.7, .7, .7])
    d = [0, t_-1]
    xlim([min(date_dt), max(date_dt)])
    xticks(date_dt[d])
    plt.gca().xaxis.set_major_formatter(myFmt)
    ylim([0, npmax(P)])
    yticks([])
    ylabel('FP')
    Ens = 'Effective Num.Scenarios =  % 3.0f'%ens[0,q]
    plt.text(date_dt[t_-1], npmax(P) - npmax(P) / 10, Ens, horizontalalignment='right',verticalalignment='bottom')
    plt.tight_layout();
    # save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

