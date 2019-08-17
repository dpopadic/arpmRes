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

# # S_EllipsoidTestTickTime [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_EllipsoidTestTickTime&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerPoissoniid).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import arange, zeros, where, cumsum, linspace
from numpy import sum as npsum

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict, date_mtop
from autocorrelation import autocorrelation
from TradeQuoteProcessing import TradeQuoteProcessing
from TradeQuoteSpreading import TradeQuoteSpreading
from InvarianceTestEllipsoid import InvarianceTestEllipsoid
# -

# ## Upload the database

try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_US_10yr_Future_quotes_and_trades'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_US_10yr_Future_quotes_and_trades'), squeeze_me=True)

# ## Process the time series, refining the raw data coming from the database

# +
quotes = struct_to_dict(db['quotes'])
trades = struct_to_dict(db['trades'])

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

t, _, q_ask, p_ask, q_bid, p_bid, t_k, _, p_last, delta_q, delta_sgn, vargout = TradeQuoteProcessing(t, dates_quotes, q_ask, p_ask, q_bid, p_bid, t_k, dates_trades, p_last, delta_q, delta_sgn, match,
                                                                                                    {1:numords_ask, 2:numords_bid})

t = t.flatten()
numords_ask, numords_bid = vargout[1], vargout[2]
q = cumsum(delta_q)  # cumulative volume of traded contracts
sgn = cumsum(delta_sgn)  # cumulative trade sign
# -

# ## Compute the realized time series of new events delta_k_t with time unit of one second

# +
i_t0 = 1  # index of window's starting time
i_t1 = len(t)  # index of window's last time
ms = (date_mtop(t[i_t1-1]) - date_mtop(t[i_t0-1])).seconds * 1000 + (date_mtop(t[i_t1-1]) - date_mtop(t[i_t0-1])).microseconds / 1000
# total len of time window expressed in wall-clock-time
t_ms = linspace(t[i_t0-1],t[i_t1-1], int(ms)+1) # time window's wall-clock-time vector expressed in milliseconds

k_0 = where(t_k[0] >= t[i_t0-1])[0][0]  # index of the first trade within the time window
k_1 = where(t_k[0] <= t[i_t1-1])[0][-1]  # index of the last trade within the time window

_, _, _, _, p_last, *_ = TradeQuoteSpreading(t_ms, t[i_t0-1:i_t1], q_ask[0,i_t0-1: i_t1], p_ask[0,i_t0-1: i_t1],
                                                  q_bid[0,i_t0-1: i_t1],p_bid[0,i_t0-1: i_t1], t_k[0,k_0:k_1+1],
                                                  p_last[0,k_0:k_1+1], q[k_0:k_1+1], sgn[k_0:k_1+1],
                                                     {1: numords_ask[0,i_t0 - 1: i_t1], 2: numords_bid[0,i_t0 - 1: i_t1]})

delta_t = 1000  # time unit of one second
t_span = arange(0,len(t_ms),delta_t)
delta_k_t = zeros((1, len(t_span) - 1))
for k in range(len(t_span) - 1):
    delta_k_t[0,k] = npsum(~np.isnan(p_last[0,t_span[k]:t_span[k + 1] - 1]))
# -

# ## Compute autocorrelations at different lags

# +
lag_ = 10

acf = autocorrelation(delta_k_t, lag_)
# -

# ## Plot the results of the IID test

# +
lag = 10  # lag to be printed
ell_scale = 2  # ellipsoid radius coefficient
fit = 3

f = figure(figsize=(12,6))
InvarianceTestEllipsoid(delta_k_t, acf[0,1:], lag, fit, ell_scale, [],'', [-5, 0]);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

