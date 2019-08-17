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

# # S_HBFPellipsoidConvergence [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_HBFPellipsoidConvergence&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerMVEStop).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import arange, r_, min as npmin, max as npmax

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, xlim, ylim, scatter, ylabel, \
    xlabel, xticks, yticks

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from HighBreakdownFP import HighBreakdownFP
from ARPM_utils import struct_to_dict, save_plot
from PlotTwoDimEllipsoid import PlotTwoDimEllipsoid
from Price2AdjustedPrice import Price2AdjustedPrice
from GarchResiduals import GarchResiduals
from BlowSpinFP import BlowSpinFP
from ColorCodedFP import ColorCodedFP
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
i_ = 2
t_ = 100

_, x_1 = Price2AdjustedPrice(StocksSPX.Date.reshape(1,-1), StocksSPX.Prices[[25],:], StocksSPX.Dividends[25])  # Cisco Systems Inc
_, x_2 = Price2AdjustedPrice(StocksSPX.Date.reshape(1,-1), StocksSPX.Prices[[5],:], StocksSPX.Dividends[5])  # General Electric
date = StocksSPX.Date[1:]

x_1 = x_1[-t_:]
x_2 = x_2[-t_:]
date = date[-t_:]
# -

# ## Compute the invariants using GARCH(1,1) fit

epsi = GarchResiduals(r_[x_1,x_2])

# ## Compute the Flexible Probability profiles using Blow-Spin method

b = 1  # number of blows
s = 0  # number of spins
p, _ = BlowSpinFP(epsi, b, s)
q_ = b + s

# ## Compute HBFP-mean and HBFP-covariance

print('Computing  HBFP-mean and HBFP-covariance')
p_tilde = 0.5
mu_HBFP, sigma2_HBFP, p_HBFP, v_HBFP, t_tilde = HighBreakdownFP(epsi, p, 0, p_tilde)

# ## Generate a static figure showing the ellipsoids computed at each iteration, as well as the volume/probability graph

# +
k_ = mu_HBFP.shape[1]

# color settings
c_vp = [0.2, 0.2, 0.6]
greyrange = arange(0,0.8,0.01)

# axis lim
c = .75
epslim1 = [min(epsi[0]) - c, max(epsi[0])+c]
epslim2 = [min(epsi[1]) - c, max(epsi[1])+c]

# figure settings
f = figure()
with plt.style.context("seaborn-whitegrid"):
    # scatter plot of observations with ellipsoid superimposed
    CM, C = ColorCodedFP(p, None, None, greyrange, 0, 1, [1, 0])
    h_1 = plt.subplot2grid((4,1),(0,0),rowspan=3)
    h_1.set_yticklabels([])
    h_1.set_xticklabels([])
    xlabel('$\epsilon_1$')
    ylabel('$\epsilon_2$')
    ell_2 = PlotTwoDimEllipsoid(mu_HBFP[:,[k_-1]], sigma2_HBFP[:,:,k_-1], 1, False, False, 'r')
    out = scatter(epsi[0, t_tilde.astype(int)], epsi[1, t_tilde.astype(int)], s=100, facecolor='none',edgecolor=[1, 0.5,0.4], marker='o', lw=1.5, zorder=10)
    for k in range(k_):
        ell_1 = PlotTwoDimEllipsoid(mu_HBFP[:,[k]], sigma2_HBFP[:,:,k], 1, False, False, [0.75, 0.75, 0.75], 0.3)
    scatter(epsi[0], epsi[1], 15, c=C, marker='.',cmap=CM)
    leg = legend(handles=[ell_2[0][0],out,ell_1[0][0]],labels=['HBFP ellipsoid','outliers','iterative ellipsoids'])
    xlim(epslim1)
    ylim(epslim2)
    plt.grid(True)
    h_2 = plt.subplot2grid((4,1),(3,0))
    h_2.set_facecolor('w')
    for k in range(k_):
        plot([p_HBFP[k], p_HBFP[k]], [v_HBFP[k], v_HBFP[k]],color=c_vp,marker='*',markersize= 3,markerfacecolor= c_vp)
    xlim([npmin(p_HBFP[1:]), npmax(p_HBFP)])
    ylim([npmin(v_HBFP) - (npmax(v_HBFP) - npmin(v_HBFP)) / 10, npmax(v_HBFP[:-1])])
    xlabel('probability')
    ylabel('volume')
    plt.grid(False)
    plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

