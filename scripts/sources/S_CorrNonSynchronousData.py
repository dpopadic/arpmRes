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

# # S_CorrNonSynchronousData [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_CorrNonSynchronousData&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerNonSyncData).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import arange, array, ones, zeros, diff, log, sqrt, min as npmin, max as npmax
from numpy import sum as npsum

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, xlim, ylim, scatter, subplots, title, xticks, yticks
import matplotlib.dates as mdates

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict, date_mtop
from intersect_matlab import intersect
# -

# ## Upload database

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_Stocks'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_Stocks'), squeeze_me=True)

SPX = struct_to_dict(db['SPX'])
KOSPI = struct_to_dict(db['KOSPI'])
# -

# ## Compute the log-prices and log-returns of the two indexes

# +
# S&P 500 (US)
NSprice = SPX.Price_close
x1 = log(NSprice)
NSdate = SPX.Date

# KOSPI (Korea)
KSprice = KOSPI.Price_close
x2 = log(KSprice)
KSdate = KOSPI.Date

# merge dataset
Date, i1, i2 = intersect(NSdate, KSdate)
logprice1 = x1[i1]
logprice2 = x2[i2]
ret1 = diff(logprice1)
ret2 = diff(logprice2)
# -

# ## Estimate the correlation concatenating the log-returns over 5 days (l=4)

# +
# concatenate the daily log-returns
l = 4
tret_ = len(ret1)

y1 = zeros((1, tret_))
y2 = zeros((1, tret_))

for t in range(l, tret_):
    y1[0,t] = sum(ret1[t - l:t])
    y2[0,t] = sum(ret2[t - l:t])

y1 = y1[[0],l:]
y2 = y2[[0],l:]

t_ = y1.shape[1]  # number of overlapping joint observations available

# compute the correlation (corr([t] is computed on the time series of y1 and y2 up to time t))
rho2 = zeros((1, t_))

for t in range(t_):
    FP = (1 / (t+1)) * ones(t+1)  # constant flexible probabilities
    y1_t = y1[0,:t+1]
    y2_t = y2[0,:t+1]
    FPstd1 = sqrt(npsum(FP * (y1_t ** 2)))
    FPstd2 = sqrt(npsum(FP * (y2_t ** 2)))
    rho2[0,t] = npsum(FP * y1_t * y2_t) / (FPstd1*FPstd2)
# -

# ## Estimate the correlation without concatenating the log-returns (l=0)

# +
y1_l0 = ret1[l:tret_]
y2_l0 = ret2[l:tret_]

rho2_l0 = zeros((1, t_))
for t in range(t_):
    FP = (1 / (t+1)) * ones(t+1)
    y1_t = y1_l0[:t+1]
    y2_t = y2_l0[:t+1]
    FPstd1 = sqrt(npsum(FP * (y1_t ** 2)))
    FPstd2 = sqrt(npsum(FP * (y2_t ** 2)))
    rho2_l0[0,t] = npsum(FP * y1_t * y2_t) / (FPstd1*FPstd2)
# -

# ## Generate the figures

# +
ln_p1 = logprice1[l + 1:]
ln_p2 = logprice2[l + 1:]
date = Date[l + 1:]

date_dt = array([date_mtop(i) for i in date])
myFmt = mdates.DateFormatter('%d-%b-%Y')
date_tick = date_dt[arange(99, len(date_dt), 200)]

# FIGURE 1: overlap 5-days (l=4)

f,ax = subplots(3,1)
# correlation
plt.sca(ax[0])
plot(date_dt, rho2[0], color='k')
xlim([min(date_dt), max(date_dt)])
xticks(date_tick)
ylim([0.1, 1])
ax[0].xaxis.set_major_formatter(myFmt)
title('Correlation')
lag = 'overlap:  % 2.0f days'%(l+1)
plt.text(date_mtop(min(date) + 100), 1, lag, horizontalalignment='left')

# l-day log-returns
plt.sca(ax[1])
scatter(date_dt, y1[0],c='b',s=10)
ax2 = ax[1].twinx()
ax2.grid(False)
ax2.scatter(date_dt, y2[0],s=10, c= [.9, .4, 0])
ylim([-0.2, 0.2])
yticks([])
xlim([min(date_dt), max(date_dt)])
xticks(date_tick)
ax[1].xaxis.set_major_formatter(myFmt)
ylim([-0.2, 0.2])
title('log-returns concatenated over %2.0f-days' % (l + 1))
ax[1].set_ylabel('SPX', color='b')
ax2.set_ylabel('KOSPI', color=[.9, .4, 0])
# log-prices
plt.sca(ax[2])
plot(date_dt, ln_p1, c='b')
ax2 = ax[2].twinx()
ax2.plot(date_dt, ln_p2, c=[.9, .4, 0])
xlim([min(date_dt), max(date_dt)])
xticks(date_tick)
ax[2].xaxis.set_major_formatter(myFmt)
title('log-prices')
ax[2].set_ylabel('SPX', color='b')
ax2.set_ylabel('KOSPI', color=[.9, .4, 0])
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

# FIGURE 2 no overlap (l=0)
# correlation
f,ax = subplots(3,1)
# correlation
plt.sca(ax[0])
plot(date_dt, rho2_l0[0], color='k')
xlim([min(date_dt), max(date_dt)])
xticks(date_tick)
ax[0].xaxis.set_major_formatter(myFmt)
ylim([0.1, 1])
title('Correlation')
plt.text(date_mtop(min(date) + 100), 1, 'No overlap', horizontalalignment='left')

# l-day log-returns
plt.sca(ax[1])
scatter(date_dt, y1_l0,c='b',s=10)
ax2 = ax[1].twinx()
ax2.scatter(date_dt, y2_l0, c= [.9, .4, 0], s=10)
ylim([-0.2, 0.2])
yticks([])
xlim([min(date_dt), max(date_dt)])
xticks(date_tick)
ax[1].xaxis.set_major_formatter(myFmt)
ylim([-0.2, 0.2])
title('log-returns')
ax[1].set_ylabel('SPX', color='b')
ax2.set_ylabel('KOSPI', color=[.9, .4, 0])

# log-prices
plt.sca(ax[2])
plot(date_dt, ln_p1, c='b')
ax2 = ax[2].twinx()
ax2.grid(False)
ax2.plot(date_dt, ln_p2, c=[.9, .4, 0])
xlim([min(date_dt), max(date_dt)])
xticks(date_tick)
ax[2].xaxis.set_major_formatter(myFmt)
title('log-prices')
ax[2].set_ylabel('SPX', color='b')
ax2.set_ylabel('KOSPI', color=[.9, .4, 0])
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

