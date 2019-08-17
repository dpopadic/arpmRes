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

# # S_EllipsoidTestWaitingTimesACDres [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_EllipsoidTestWaitingTimesACDres&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=IIDHFACDdTres).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import where, diff, linspace

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict, date_mtop
from autocorrelation import autocorrelation
from TradeQuoteProcessing import TradeQuoteProcessing
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

dates_trades = trades.time_names  #
t_k = trades.time  # time vector of trades
p_last = trades.price  # last transaction prices
delta_q = trades.siz  # flow of traded contracts' volumes
delta_sgn = trades.aggress  # trade sign flow
match = trades.mtch  # match events: - the "1" value indicates the "start of a match event" while zeros indicates the "continuation of a match event"
#              - the db is ordered such that the start of a match event is in the last column corresponding to that event

t, _, _, _, _, _, t_k, _, _, _, _, _ = TradeQuoteProcessing(t, dates_quotes, q_ask, p_ask, q_bid, p_bid, t_k, dates_trades,
                                                         p_last, delta_q, delta_sgn, match)
t = t.flatten()
t_k = t_k.flatten()
# ## Compute the gaps between subsequent events

k_0 = where(t_k >= t[0])[0][0]    # index of the first trade within the time window
k_1 = where(t_k <= t[-1])[0][-1]  # index of the last trade within the time window
ms = (date_mtop(t_k[k_1]) - date_mtop(t_k[k_0])).seconds * 1000 + (date_mtop(t_k[k_1]) - date_mtop(t_k[k_0])).microseconds / 1000
t_k = linspace(t_k[k_0],t_k[k_1], int(ms)) # time window's wall-clock-time vector expressed in milliseconds
delta_t_k = diff(t_k)  # waiting times
# -

# ## ACD fit (Requires the external package ACD_Models_FEX)

# +
q = 1  # maximum lag for the duration
p = 1  # maximum lag for the volatility
stdMethod = 1

tmp_dt_n = [0, delta_t_k]
specOut = ACD_Fit(tmp_dt_n.T,'exp', q, p, stdMethod)  # fitting
# estimated parameters
c = specOut.w
b = specOut.p
a = specOut.q
# estimated sigma_n
sigma_n = specOut.h.T

# residuals
ACD_epsi = delta_t_k / sigma_n[1:]
# -

# ## Compute autocorrelations at different lags

lag_ = 10
acf = autocorrelation(ACD_epsi, lag_)

# ## Plot the results of the IID test

# +
lag = 10  # lag to be printed
ell_scale = 1.6  # ellipsoid radius scale
fit = 2  # exponential fit

f = figure(figsize=(12,6))
InvarianceTestEllipsoid(delta_t_k, acf[0,1:], lag_, fit, ell_scale, [],
                        'Invariance test on the residuals of an ACD fit on arrival times', [-4, 19]);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

