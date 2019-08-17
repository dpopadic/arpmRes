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

# # S_StochasticVolatilityLeverage [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_StochasticVolatilityLeverage&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerFig098StochVol).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import arange, array, zeros, diff, cov, mean, log, exp, sqrt, r_
from numpy import sum as npsum, min as npmin, max as npmax, mean as npmean

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, scatter, ylabel, \
    xlabel, title, xticks, yticks
import matplotlib.dates as mdates

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict, date_mtop
from FitStochasticVolatilityModel import FitStochasticVolatilityModel
from FilterStochasticVolatility import FilterStochasticVolatility
from PlotTwoDimEllipsoid import PlotTwoDimEllipsoid
# -

# ## Upload database

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_Stocks'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_Stocks'), squeeze_me=True)

SPX = struct_to_dict(db['SPX'])

# daily prices and log-prices
price = SPX.Price_close
date = SPX.Date
logprice = log(price)
# -

# ## Compute weekly prices, returns and the log-square volatility
# ##pick weekly data

# +
w = arange(0, len(logprice), 5)
date = array([date_mtop(i) for i in SPX.Date[w]])

# prices
price = price[w]

# log-prices
logprice_weekly = log(price)

# log-returns
ret = diff(logprice_weekly)

# y = log(squared returns)
y = log(ret ** 2)
# -

# ## Fit the stochastic volatility model

# +
# initial parameters
phi0 = 0
phi1 = .99
sQ = 0.14
alpha = npmean(y)
sR0 = 0.9
mu1 = -2
sR1 = 2
initpar = [phi0, phi1, sQ, alpha, sR0, mu1, sR1]

param, fval, exitflag, output = FitStochasticVolatilityModel(y, initpar)
phi = param[0]
phi1 = param[1]
sQ = param[2]
alpha = param[3]
sR0 = param[4]
mu1 = param[5]
sR1 = param[6]
_, log_hiddenvol2 = FilterStochasticVolatility(y, phi0, phi1, sQ, alpha, sR0, mu1, sR1)

# hidden volatility
hidden_vol = sqrt(exp((log_hiddenvol2)))
# -

# ## Compute the daily intra-week empirical volatility

# +
t_ = len(w) - 1  # lenght of the time-series
empirical_vol = zeros((1, t_))

for index in range(t_):
    empirical_vol[0,index] = 0.2 * sqrt(npsum(diff(logprice[w[index]:w[index + 1] - 1]) ** 2))
# -

# ## Compute location and dispersion needed to plot ellipsoid in the (log-ret vs empirical vol. scatter plot)

ret_vol = r_[ret.reshape(1,-1), empirical_vol]
m = mean(ret_vol, 1, keepdims=True)
s2 = cov(ret_vol)

# ## Generate the figure

# +
date_tick = arange(0, t_, 80)  # tick for the time axes
xticklabels = date[date_tick[::2]]

# colors
c0 = [0.9, 0.5, 0]
c1 = [.4, .4, 1]
c2 = [0.3, 0.3, 0.3]
myFmt = mdates.DateFormatter('%d-%b-%y')

f = figure(figsize=(12,6))

# axes for prices
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax1.set_facecolor('white')
plt.axis([min(date), max(date), npmin(price), npmax(price) + 5])
ylabel('prices', color=c1)
ax1.plot(date[1:], price[1:], color=c1)  # prices
ax1.set_xticks(xticklabels)
ax1.xaxis.set_major_formatter(myFmt)
ax1.tick_params(axis='y', colors=c1)

# axes for log-returns
ax2 = ax1.twinx()
ax2.grid(False)
ax2.scatter(date[1:], ret, s=2.5, c=c2, marker='.')  # log-returns
ax2.set_ylabel('log-returns', color=c2)
ax2.tick_params(axis='y', colors=c2)

# axes for hidden volatility
ax3 = plt.subplot2grid((2, 2), (1, 0))
ax3.set_facecolor('white')
plt.axis([min(date), max(date), npmin(hidden_vol), npmax(hidden_vol)])
ylabel('hidden vol',color=c1)
title('VOLATILITY')
ax3.plot(date[1:], hidden_vol, color=c1)  # hidden volatility
ax3.grid(False)
ax3.set_xticks(xticklabels)
ax3.xaxis.set_major_formatter(myFmt)
ax3.tick_params(axis='y', colors=c1)

# axes for empirical volatility
ax4 = ax3.twinx()
ax4.grid(False)
ax4.set_ylabel('empirical vol.', color=c0)
ax4.plot(date[1:], empirical_vol.flatten(), color=c0, lw=1.3)  # empirical volatility
ax4.tick_params(axis='y', colors=c0)

# axes for the scatter plot (leverage effect)
ax5 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)

plt.axis([npmin(ret), npmax(ret), 0, npmax(empirical_vol)])
ylabel('empirical vol.')
xlabel('log-returns')
title('LEVERAGE EFFECT')
scatter(ret, empirical_vol, 3, c2, '*')
PlotTwoDimEllipsoid(m, s2, 1, 0, 0, c0, 2, fig=plt.gcf())
plt.axis('auto')
plt.tight_layout();
plt.show()
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
