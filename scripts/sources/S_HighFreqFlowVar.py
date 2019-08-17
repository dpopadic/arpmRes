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

# # S_HighFreqFlowVar [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_HighFreqFlowVar&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=exer-mkt-micro-structure-copy-1).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

from matplotlib.ticker import FormatStrFormatter

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import array, zeros, where, cumsum, linspace
from numpy import min as npmin, max as npmax

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, xlim, ylim, ylabel, \
    title, xticks, yticks, subplots
import matplotlib.dates as mdates

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import date_mtop, struct_to_dict, save_plot
from TradeQuoteProcessing import TradeQuoteProcessing
from TradeQuoteSpreading import TradeQuoteSpreading
# -

# ## Upload the data from db_US_10yr_Future_quotes_and_trades

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_US_10yr_Future_quotes_and_trades'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_US_10yr_Future_quotes_and_trades'), squeeze_me=True)

quotes = struct_to_dict(db['quotes'])
trades = struct_to_dict(db['trades'])
# -

# ## Process the time series so that variables are defined at each closk-time corresponding to the occurrence of the generic k-th event, using function TradeQuoteProcessing

# +
dates_quotes = quotes.time_names  #
t = quotes.time  # time vector of quotes
p_bid = quotes.bid  # best bids
p_ask = quotes.ask  # best asks
q_bid = quotes.bsiz  # bid sizes
q_ask = quotes.asiz  # ask sizes
numords_bid = quotes.bnumords  # number of separate limit orders on bid
numords_ask = quotes.anumords  # number of separate limit orders on ask

dates_trades = trades.time_names
t_k = trades.time  # time vector of trades
p_last = trades.price  # last transaction values
delta_q = trades.siz  # flow of traded contracts' sizes
delta_sgn = trades.aggress  # trade sign flow
match = trades.mtch  # match events: - the "1" value indicates the "start of a match event" while zeros indicates the "continuation of a match event"
#              - the db is ordered such that the start of a match event is in the last column corresponding to that event

t, _, q_ask, p_ask, q_bid, p_bid, t_k, _, p_last, delta_q, delta_sgn, vargout = \
    TradeQuoteProcessing(t, dates_quotes, q_ask, p_ask, q_bid, p_bid, t_k, dates_trades, p_last, delta_q, delta_sgn, match,
                         {1:numords_ask, 2:numords_bid})

t = t.flatten()
numords_ask, numords_bid = vargout[1], vargout[2]
q = cumsum(delta_q)  # cumulative volume of traded contracts
sgn = cumsum(delta_sgn)  # cumulative trade sign
# -

# ## Compute the time series of the cumulative volume and the cumulative sign as functions of wall clock time using function TradeQuoteSpreading

# +
i_t0 = 1  # index of window's starting time
i_t1 = 1250  # index of window's last time
ms = (date_mtop(t[i_t1-1]) - date_mtop(t[i_t0-1])).seconds * 1000 + (date_mtop(t[i_t1-1]) - date_mtop(t[i_t0-1])).microseconds / 1000+1
# total len of time window expressed in wall-clock-time
t_ms = linspace(t[i_t0-1],t[i_t1-1], int(ms)+1) # time window's wall-clock-time vector expressed in milliseconds

k_0 = where(t_k[0] >= t[i_t0])[0][0]  # index of the first trade within the time window
k_1 = where(t_k[0] <= t[i_t1])[0][-1]  # index of the last trade within the time window

_, _, _, _, p_last, q, sgn,_ = TradeQuoteSpreading(t_ms, t[i_t0-1:i_t1], q_ask[0,i_t0-1:i_t1], p_ask[0,i_t0-1:i_t1],
                                                               q_bid[0,i_t0-1:i_t1], p_bid[0,i_t0-1:i_t1], t_k[0,k_0:k_1+1],
                                                               p_last[0,k_0:k_1+1], q[k_0:k_1+1], sgn[k_0:k_1+1])

q_line = zeros(q.shape)
sgn_line = zeros(sgn.shape)

# fill q and sgn where zeros (aimed at plotting lines)
if np.isnan(p_last[0,0]):
    if k_0 > 0:
        q_line[0,0] = q[0,k_0 - 1]
        sgn_line[0,0] = sgn[0,k_0 - 1]
    else:
        q_line[0,0] = q[0,0]
        sgn_line[0,0] = sgn[0,0]

for i in range(1,len(t_ms)):
    if sgn[0,i]==0:
        sgn_line[0,i] = sgn_line[0,i - 1]
        q_line[0,i] = q_line[0,i - 1]
    else:
        sgn_line[0,i] = sgn[0,i]
        q_line[0,i] = q[0,i]
# -

# ## Generate a figure showing the plot of the cumulative volume and the cumulative sign

# +
# color settings
orange = [.9, .3, .0]
blue = [0, 0, .8]

t_ms_dt = array([date_mtop(i) for i in t_ms])
xtick = linspace(1999, len(t_ms_dt)-1, 8, dtype=int)
myFmt = mdates.DateFormatter('%H:%M:%S')

# axes settings
ymax_2 = npmax(q_line) + 5
ymin_2 = npmin(q_line[0,q_line[0]>0])
ytick_2 = linspace(ymin_2,ymax_2,5)
ymax_3 = npmax(sgn_line) + 1
ymin_3 = npmin(sgn_line) - 1
ytick_3 = linspace(ymin_3,ymax_3, 5)

f, ax = subplots(1,1)
plt.sca(ax)
ax.xaxis.set_major_formatter(myFmt)
ylabel('Cumulative volume',color=orange)
ylim([ymin_2, ymax_2])
idx = q[0] > 0
plt.scatter(t_ms_dt[idx], q[0, idx], color=orange, marker='.', s=2)
plot(t_ms_dt, q_line[0], color=orange, lw=1)
ax2 = ax.twinx()
ylim([ymin_3, ymax_3])
yticks(ytick_3)
plt.sca(ax2)
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
plt.grid(False)
title('Flow variables for US 10yr Future: %s' % t_ms_dt[0].strftime('%Y-%b-%d'))
ylabel('Cumulative sign',color=blue)
idx = sgn[0]!=0
plt.scatter(t_ms_dt[idx], sgn[0,idx], color=blue, marker='.',s=2)
ax2.set_xticks(t_ms_dt[xtick])
ax.set_xlim([min(t_ms_dt), max(t_ms_dt)])
ax.set_yticks(ytick_2)
plot(t_ms_dt, sgn_line[0], color=blue, lw=1);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
