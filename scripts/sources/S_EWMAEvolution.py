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

# # S_EWMAEvolution [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_EWMAEvolution&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-ewmanum-ex).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import arange, array, zeros, sort, argsort, cumsum, log, exp, sqrt
from numpy import sum as npsum

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, xlim
import matplotlib.dates as mdates

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict, date_mtop
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
tauHL = 25
c = 0.05
w = 200  # trailing window
t_start = 252
# -

# ## Set the exponential decay probabilities with half-life of 25 days and using a trailing window of 200 observations

# +
p = exp(-log(2) / tauHL*arange(w - 1, 0 + -1, -1)).reshape(1,-1)
gamma_w = npsum(p)
p = p / gamma_w

# time series of log-returns
epsi = log(v_SP[1:] / v_SP[:-1])

ewma = zeros(t_ - t_start + 1)
ewm_sd = zeros(t_ - t_start + 1)
ewm_q = zeros(t_ - t_start + 1)
# -

# ## Compute the time series of: the EWMA the EWM standard deviation the EWM quantile at confidence level c = 5#

# +
for t in range(t_start,t_):  # scenarios
    x = epsi[t - w:t]

    # ewma
    ewma[t - t_start] = p@x.T

    # ewm standard deviation
    ewm_sd[t - t_start] = sqrt(p@(x.T) ** 2 - ewma[t - t_start] ** 2)

    # ewm quantile
    q, t_sort = sort(x), argsort(x)
    ewm_q[t - t_start] = min(q[cumsum(p[0,t_sort]) >= c])  # ## Draw the plot

dates_dt = array([date_mtop(i) for i in dates])
myFmt = mdates.DateFormatter('%d-%b-%Y')

figure()
plot(dates_dt[t_start:], epsi[t_start-1:], '.b')
plot(dates_dt[t_start:], ewma, '-g')
plot(dates_dt[t_start:], ewma + 2*ewm_sd, '-r', lw=1)
plot(dates_dt[t_start:], ewma - 2*ewm_sd, '-r', lw=1)
plot(dates_dt[t_start:], ewm_q, '-k', lw=1)
xlim([dates_dt[t_start], dates_dt[-1]])
plt.gca().xaxis.set_major_formatter(myFmt);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
