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

# # S_HighFreqVolumeTime [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_HighFreqVolumeTime&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerVolEvol).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import arange, array, zeros, where, cumsum, interp, linspace, abs
from numpy import min as npmin, max as npmax

from scipy.io import loadmat, savemat

import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict, date_mtop
from TradeQuoteProcessing import TradeQuoteProcessing
from TradeQuoteSpreading import TradeQuoteSpreading
# -

# ## Parameters

i_t0 = 1  # index of window's starting time

# ## Upload data from db_US_10yr_Future_quotes_and_trades

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_US_10yr_Future_quotes_and_trades'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_US_10yr_Future_quotes_and_trades'), squeeze_me=True)

quotes = struct_to_dict(db['quotes'])
trades = struct_to_dict(db['trades'])
# -

# ## Process the time series so that variables are defined at each occurrence time, using function TradeQuoteProcessing

# +
dates_quotes = quotes.time_names  #
t = quotes.time  # time vector of quotes
p_bid = quotes.bid  # bid prices
p_ask = quotes.ask  # ask prices
q_bid = quotes.bsiz  # bid volumes
q_ask = quotes.asiz  # ask volumes
numords_bid = quotes.bnumords  # number of separate limit orders on bid
numords_ask = quotes.anumords  # number of separate limit orders on ask

dates_trades = trades.time_names  #
t_k = trades.time  # time vector of trades
p_last = trades.price  # last transaction prices
delta_q = trades.siz  # flow of traded contracts' volumes
delta_sgn = trades.aggress  # trade sign flow
match = trades.mtch  # match events: - the "1" value indicates the "start of a match event" while zeros indicates the "continuation of a match event"
#              - the db is ordered such that the start of a match event is in the last column corresponding to that event

t, dates_quotes, q_ask, p_ask, q_bid, p_bid, t_k, dates_trades, p_last, delta_q, delta_sgn, vargout = \
    TradeQuoteProcessing(t, dates_quotes, q_ask, p_ask, q_bid, p_bid, t_k, dates_trades, p_last, delta_q, delta_sgn, match,
                         {1:numords_ask, 2:numords_bid})
t = t.flatten()
numords_ask, numords_bid = vargout[1], vargout[2]
q = cumsum(delta_q)  # cumulative volume of traded contracts
sgn = cumsum(delta_sgn)  # cumulative trade sign
# -

# ## Compute microprice and total exchanged volume as functions of wall clock time and volume time

# +
i_t1 = len(t)  # index of window's last time
ms = (date_mtop(t[i_t1-1]) - date_mtop(t[i_t0-1])).seconds * 1000 + (date_mtop(t[i_t1-1]) - date_mtop(t[i_t0-1])).microseconds / 1000  # total len of time window expressed in wall-clock-time
t_ms = linspace(t[i_t0-1],t[i_t1-1], int(ms)+1)  # time window's wall-clock-time vector expressed in milliseconds

k_0 = where(t_k[0] >= t[i_t0-1])[0][0]  # index of the first trade within the time window
k_1 = where(t_k[0] <= t[i_t1-1])[0][-1]  # index of the last trade within the time window

q_ask, p_ask, q_bid, p_bid, _, q_t, _, _ = TradeQuoteSpreading(t_ms, t[i_t0-1:i_t1], q_ask[0,i_t0-1: i_t1], p_ask[0,i_t0-1: i_t1],
                                                  q_bid[0,i_t0-1: i_t1],p_bid[0,i_t0-1: i_t1], t_k[0,k_0:k_1+1],
                                                  p_last[0,k_0:k_1+1], q[k_0:k_1+1], sgn[k_0:k_1+1])

p_mic = (p_bid * q_ask + p_ask * q_bid) / (q_ask + q_bid)  # microprice as a function of wall clock time
delta_a = 23  # width of activity time bins
a_t = arange(npmin(q_t),npmax(q_t)+delta_a,delta_a)  # vector of volume times
t_a = interp(a_t, q, t_k.flatten())  # vector of wall clock time as a function of volume time
p_mic_a = interp(t_a, t_ms, p_mic.flatten())  # time changed microprice, i.e. microprice as a function of volume time

# fill q_t where zeros (aimed at plotting lines)
q_t_line = zeros((1, len(t_ms)))
index = where(q_t[0]!=0)[0]
if k_0 > 1:
    q_t_line[0,index[0] - 1]=q_t[k_0 - 1]

for k in range(len(index) - 1):
    q_t_line[0, index[k]: index[k + 1] - 1] = q_t[0,index[k]]

q_t_line[index[-1]:] = q_t[0,index[-1]]

vars_to_save = {varname: var for varname, var in locals().items() if isinstance(var,(np.ndarray,np.float,np.int))}
vars_to_save.update({'dates_quotes': dates_quotes, 'dates_trades': dates_trades, 'quotes': quotes, 'trades':trades})

savemat(os.path.join(TEMPORARY_DB,'db_HighFreqVolumeTime'),vars_to_save)
# -

# ## Generate a figure showing the microprice and the total exchanged volume as functions of wall clock time and volume time

# axes settings
timegrid = [date_mtop(i) for i in linspace(t_ms[0],t_ms[-1], 3)]
pgrid_min = np.nanmin(p_mic)
pgrid_max = np.nanmax(p_mic)
pgrid = linspace(pgrid_min,pgrid_max,5)
volgrid_min = np.nanmin(q_t[0,q_t[0]>0]) - 1
volgrid_max = np.nanmax(q_t[0,q_t[0]>0]) + 1
volgrid = linspace(volgrid_min, volgrid_max, 3)
myFmt = mdates.DateFormatter('%H:%M:%S')
t_ms_dt = array([date_mtop(i) for i in t_ms])
f, ax  = subplots(2,2)
ax[0,0].plot(t_ms_dt,p_mic[0],c='r',lw=1)
ax[0,0].set_xticks(timegrid)
ax[0,0].set_yticks(pgrid)
ax[0,0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax[0,0].xaxis.set_major_formatter(myFmt)
ax[0,0].axis([min(t_ms_dt), max(t_ms_dt), pgrid_min, pgrid_max])
ax[0,0].set_ylabel('Microprice')
ax[0,0].set_xlabel('Wall Clock Time')
ax[0,0].set_title('Time evolution')
plt.grid(True)
# right-top plot
ax[0,1].set_xticks(volgrid)
ax[0,1].set_yticks(pgrid)
ax[0,1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax[0,1].axis([volgrid_min, volgrid_max, pgrid_min, pgrid_max])
ax[0,1].set_ylabel('Microprice')
ax[0,1].set_xlabel('Volume Time')
ax[0,1].plot(a_t, p_mic_a, lw=1, color='r')
ax[0,1].set_title('Volume Time Activity Evolution')
plt.grid(True)
# left-bottom plot
ax[1,0].set_xticks(timegrid)
ax[1,0].set_yticks(volgrid)
ax[1,0].axis([min(t_ms_dt), max(t_ms_dt), volgrid_min, volgrid_max])
ax[1,0].xaxis.set_major_formatter(myFmt)
ax[1,0].set_ylabel('Exchanged Volume')
ax[1,0].set_xlabel('Wall Clock Time')
index = where(q_t[0]!=0)[0]
ax[1,0].scatter(t_ms_dt[index], q_t[0,index], marker='.',s=5,color='b')
for k1,k2 in zip(index[:-1],index[1:]):
    ax[1, 0].plot([t_ms_dt[k1],t_ms_dt[k2]], [q_t[0,k1],q_t[0,k1]], lw=1, color='b')
plt.grid(True)
# right-bottom plot
ax[1,1].axis([volgrid_min, volgrid_max, volgrid_min, volgrid_max])
ax[1,1].set_ylabel('Exchanged Volume')
ax[1,1].set_xlabel('Volume Time')
ax[1,1].plot(a_t, a_t, lw=1,color='b')
ax[1,1].set_xticks(volgrid)
ax[1,1].set_yticks(volgrid)
plt.grid(True)
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
