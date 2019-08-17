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

# # S_BFEWMAEvolution [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_BFEWMAEvolution&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-ewmanum-ex-copy-1).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import arange, array, zeros, sort, argsort, cumsum, abs, log, exp, sqrt
from numpy import sum as npsum, min as npmin

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, xlim
import matplotlib.dates as mdates

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import struct_to_dict, save_plot, date_mtop
# -

# ## Upload the database

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_StocksS_P'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_StocksS_P'), squeeze_me=True)

Data = struct_to_dict(db['Data'])
# settings
v_SP = Data.SP_index
dates = Data.Dates
t_ = len(v_SP) - 1
t_start = 1  # starting point of the strategy
tauHL = 25
c = 0.05

# time series of log-returns
epsi = log(v_SP[1:] / v_SP[:-1])
epsi = epsi[t_start-1:]
epsi = epsi.reshape(1,-1)

two_ewma = zeros(t_)
two_ewm_sd = zeros(t_)
two_ewm_q = zeros(t_)
for t in range(t_):
    # ## Set the backward/forward exponential decay probabilities with half-life 25 days
    p = exp(-log(2) / tauHL*abs(t - arange(t_))).reshape(1,-1)
    gamma_t = npsum(p)
    p = p / gamma_t

    # ## Compute the time series of: the back/for-ward EWMA the back/for-ward EWM standard deviation the back/for-ward EWM quantile at confidence level c = 5#
    # ewma
    two_ewma[t] = p@epsi.T

    # ewm standard deviation
    two_ewm_sd[t] = sqrt(p@(epsi.T ** 2))

    # ewm quantile
    q, t_sort = sort(epsi), argsort(epsi)
    two_ewm_q[t] = npmin(q[0,cumsum(p[0,t_sort]) >= c])
# -

# ## Draw the plot

# +
dates_dt = array([date_mtop(i) for i in dates])
myFmt = mdates.DateFormatter('%d-%b-%Y')

figure()
plot(dates[1:], epsi[0], '.b')
plot(dates[1:], two_ewma, '-g')
plot(dates[1:], two_ewma + 2 * two_ewm_sd, '-r')
plot(dates[1:], two_ewma - 2 * two_ewm_sd, '-r')
plot(dates[1:], two_ewm_q, '-k')
xlim([dates[0], dates[-1]])
plt.gca().xaxis.set_major_formatter(myFmt)
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

