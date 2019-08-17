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

# # S_UnconditionalDistribution [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_UnconditionalDistribution&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ex-uncond-distrib-p-and-l).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import arange, array, linspace, round, log, exp
from numpy import sum as npsum, min as npmin, max as npmax

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, bar, xlim, ylim, scatter, ylabel, \
    title
import matplotlib.dates as mdates

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, date_mtop
from HistogramFP import HistogramFP
from GarchResiduals import GarchResiduals
from Stats import Stats
from ColorCodedFP import ColorCodedFP
# -

# ## Upload database

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_MomStratPL'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_MomStratPL'), squeeze_me=True)

dailypnl = db['dailypnl']
dates = db['dates']
# -

# ## Select data and set flexible probabilities

# +
y = dailypnl  # select observations
t_ = len(dates)

lam = log(2) / 180
p = exp(-lam *arange(len(y),0,-1)).reshape(1,-1)
p = p /npsum(p)  # FP-profile: exponential decay 6 months
ens = exp(npsum(-p*log(p)))  # effective number of scenarios
# -

# ## Compute the invariants using GARCH(1,1) fit

epsi = GarchResiduals(y[np.newaxis,...], t_, p)

# ## Compute statistics

#y
mu_y, sdev_y, VaR_y, CVaR_y, skewness_y, kurtosis_y = Stats(y[np.newaxis,...], p)
# epsi
mu_e, sdev_e, VaR_e, CVaR_e, skewness_e, kurtosis_e = Stats(epsi, p)

# ## Generate figures

# +
option = namedtuple('option', 'n_bins')

option.n_bins = round(7*log(ens))
hgram, x_hgram = HistogramFP(y[np.newaxis,...], p, option)
x_m = npmin(x_hgram)
x_M = npmax(x_hgram)
x_mM = x_M - x_m
hgram_1, x_hgram_1 = HistogramFP(epsi, p, option)
x1_m = npmin(x_hgram_1)
x1_M = npmax(x_hgram_1)
x1_mM = x1_M - x1_m

e_m = npmin(epsi)
e_M = npmax(epsi)
e_mM = e_M - e_m
y_m = npmin(y)
y_M = npmax(y)
y_mM = y_M - y_m
d = linspace(0,t_-1,4, dtype=int)
dates_dt = array([date_mtop(i) for i in dates])

f = figure()
CM, C = ColorCodedFP(p, None, None, arange(0,0.81,0.01), 0, 18, [12, 3])

myFmt = mdates.DateFormatter('%d-%b-%Y')
ax = plt.subplot2grid((3,3),(0,0),colspan=2)
plt.sca(ax)
# Flexible Probability profile
wid = dates_dt[1].toordinal()-dates_dt[0].toordinal()
b = bar(dates_dt, p[0], width=wid, facecolor=[.7, .7, .7], edgecolor=[.7, .7, .7])
xlim([min(dates_dt), max(dates_dt)])
plt.xticks(dates_dt[d])
ax.xaxis.set_major_formatter(myFmt)
ylim([0, npmax(p)])
plt.yticks([])
ylabel('FP')
ensT = 'Effective Num.Scenarios =  %3.0f'%ens
plt.text(0.05, 0.8, ensT, horizontalalignment='left',verticalalignment='bottom',transform=ax.transAxes)
title('Flexible Probabilities - Exponential Decay');

# invariants
ax = plt.subplot2grid((3,3),(1,2))
plt.sca(ax)
b_1 = plt.barh(x_hgram_1[:-1], hgram_1[0], height=x_hgram_1[1]-x_hgram_1[0] , facecolor=[.7, .7, .7], edgecolor=[.3, .3, .3])
plt.axis([0, npmax(hgram_1) + (npmax(hgram_1) / 20), x1_m - .15*x1_mM, x1_M + .15*x1_mM])
plt.xticks([])
plt.yticks([])
stat1 = 'Mean  % 1.3e \nSdev    %1.3e \nVaR      %1.3e \nCVaR   %1.3e \nSkew   %1.3e \nKurt     %1.3e'\
        %(mu_e,sdev_e,VaR_e,CVaR_e,skewness_e,kurtosis_e)
plt.text(0.4, 0.75, stat1,horizontalalignment='left',verticalalignment='bottom', transform=ax.transAxes)

ax = plt.subplot2grid((3,3),(1,0),colspan=2)
plt.sca(ax)
ax.set_facecolor('white')
scatter(dates_dt, epsi, 10, c=C, marker='.',cmap=CM)
xlim([min(dates_dt), max(dates_dt)])
plt.xticks(dates_dt[d])
ax.xaxis.set_major_formatter(myFmt)
title('GARCH residuals')

# P&L
ax = plt.subplot2grid((3,3),(2,2))
plt.sca(ax)
plt.barh(x_hgram[:-1], hgram[0],height=x_hgram[1]-x_hgram[0], facecolor=[.7, .7, .7], edgecolor=[.3, .3, .3])
plt.axis([0, npmax(hgram) + (npmax(hgram) / 20), x_m - .15*x_mM, x_M + .15*x_mM])
plt.xticks([])
plt.yticks([])
stat1 = 'Mean  % 1.3e \nSdev    %1.3e \nVaR      %1.3e \nCVaR   %1.3e \nSkew   %1.3e \nKurt     %1.3e'\
        %(mu_y,sdev_y,VaR_y,CVaR_y,skewness_y,kurtosis_y)
plt.text(0.4, 0.75, stat1,horizontalalignment='left',verticalalignment='bottom', transform=ax.transAxes)

# colormap(CM)
ax = plt.subplot2grid((3,3),(2,0),colspan=2)
plt.sca(ax)
ax.set_facecolor('white')
scatter(dates_dt, y, 10, c=C, marker='.',cmap=CM)
ylim([y_m - .15*y_mM, y_M + .15*y_mM])
xlim([min(dates_dt), max(dates_dt)])
plt.xticks(dates_dt[d])
ax.xaxis.set_major_formatter(myFmt)
title('P&L realizations')
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
