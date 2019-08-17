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

# # S_ImpliedLeverageEffect [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_ImpliedLeverageEffect&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExImplVolLeverageEff).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import arange, array, diff, log, exp, r_
from numpy import sum as npsum, min as npmin, max as npmax

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, scatter, ylabel, \
    xlabel, title, xticks, yticks
import matplotlib.dates as mdates

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict, date_mtop
from FPmeancov import FPmeancov
from PlotTwoDimEllipsoid import PlotTwoDimEllipsoid
# -

# ## Upload data from db_ImpliedVol_SPX

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_ImpliedVol_SPX'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_ImpliedVol_SPX'), squeeze_me=True)  # implied volatility surface for SP500

db_ImpliedVol_SPX = struct_to_dict(db['db_ImpliedVol_SPX'])

tau = db_ImpliedVol_SPX.TimeToMaturity
delta = db_ImpliedVol_SPX.Delta  # delta-moneyness
sigma_delta = db_ImpliedVol_SPX.Sigma

implied_vol = sigma_delta[0, delta == 0.5, 1:]  # at the money option expiring in tau[0] years
prices = db_ImpliedVol_SPX.Underlying
logrets = diff(log(prices))
dates = db_ImpliedVol_SPX.Dates[1:]
dates = array([date_mtop(i) for i in dates])

t_ = len(dates)

lam = log(2) / 90  # exp decay probs, half life 3 months
FP = exp(-lam * arange(t_, 1 + -1, -1))
FP = (FP / npsum(FP)).reshape(1,-1)

m, s2 = FPmeancov(r_[logrets[np.newaxis,...], implied_vol], FP)

# colors
c0 = [.9, .4, 0]
c1 = [.4, .4, 1]
c2 = [0.3, 0.3, 0.3]
myFmt = mdates.DateFormatter('%d-%b-%y')
# -

# ## Generate the figure

# +
date_tick = range(0,t_,150)  # tick for the time axes
xticklabels = dates[date_tick]  # labels for dates

f = figure()

# axes for prices
ax1 = plt.subplot2grid((2, 5), (0, 0), colspan=3)
ax1.plot(dates, prices[1:], color=c1)  # prices
ax1.set_xticks(xticklabels)
ax1.xaxis.set_major_formatter(myFmt)
ax1.tick_params(axis='y', colors=c1)
ylabel('prices', color=c1)

# axes for log-returns
ax2 = ax1.twinx()
ax2.scatter(dates, logrets, s=2.5, c=c2, marker='.')  # log-returns
ax2.set_ylabel('log-returns', color=c2)
ax2.tick_params(axis='y', colors=c2)
ax1.axis([min(dates), max(dates), npmin(prices), npmax(prices) + 5])

# axes for hidden volatility
ax3 = plt.subplot2grid((2, 5), (1, 0), colspan=3)
plt.axis([min(dates), max(dates), npmin(implied_vol), npmax(implied_vol)])
ylabel('hidden vol',color=c1)
title('VOLATILITY')
ax3.plot(dates, implied_vol.flatten(), color=c1)  # hidden volatility
ax3.set_xticks(xticklabels)
ax3.xaxis.set_major_formatter(myFmt)
ax3.tick_params(axis='y', colors=c1)

# axes for the scatter plot (leverage effect)
ax4 = plt.subplot2grid((2, 5), (0, 3), colspan=2, rowspan=2)
plt.axis([npmin(logrets), npmax(logrets), 0.8*npmin(implied_vol), 1.3*npmax(implied_vol)])
ylabel('implied vol.')
xlabel('log-returns')
title('LEVERAGE EFFECT')
scatter(logrets, implied_vol.flatten(), 3, c2, '*')
PlotTwoDimEllipsoid(m, s2, 1, 0, 0, c0, 2, fig=plt.gcf())
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
