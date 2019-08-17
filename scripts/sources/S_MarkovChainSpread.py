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

# # S_MarkovChainSpread [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_MarkovChainSpread&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=MarkovTPmic).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import maximum, unique, zeros

from scipy.io import loadmat

import matplotlib.pyplot as plt

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import struct_to_dict
from TradeQuoteProcessing import TradeQuoteProcessing
from MatchTime import MatchTime

# parameter
k = 0.01
# -

# ## Upload the database

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_US_10yr_Future_quotes_and_trades'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_US_10yr_Future_quotes_and_trades'), squeeze_me=True)

quotes = struct_to_dict(db['quotes'])
trades = struct_to_dict(db['trades'])
# -

# ## Process the time series, refining the raw data coming from the database

# +
dates_quotes = quotes.time_names  #
t = quotes.time  # time vector of quotes
p_bid = quotes.bid  # bid prices
p_ask = quotes.ask  # ask prices
q_bid = quotes.bsiz  # bid volumes
q_ask = quotes.asiz  # ask volumes

dates_trades = trades.time_names  #
t_n = trades.time  # time vector of trades
p_last = trades.price  # last transaction prices
delta_q = trades.siz  # flow of traded contracts' volumes
delta_sgn = trades.aggress  # trade sign flow
match = trades.mtch  # match events: - the "1" value indicates the "start of a match event" while zeros indicates the "continuation of a match event"
#              - the db is ordered such that the start of a match event is in the last column corresponding to that event

t, _, _, p_ask, _, p_bid, t_n, _, _, _, _,_ = TradeQuoteProcessing(t, dates_quotes, q_ask, p_ask, q_bid, p_bid, t_n,
                                                                 dates_trades, p_last, delta_q, delta_sgn, match)
# -

# ## Compute the spread only at trade times
s = p_ask - p_bid
s, _ = MatchTime(s, t, t_n)

# +
# ## Compute the tick size and the transition matrix

s_u = unique(s)
gamma = s_u[1] - s_u[0]  # tick size

s_1 = s[:-1]
s_2 = s[1:]
# transition matrix
p = zeros((2,2))
p[0, 0] = sum((s_1 == s_u[0]) & (s_2 == s_u[0])) / sum(s_1 == s_u[0])
p[0, 1] = sum((s_1 == s_u[0]) & (s_2 == s_u[1])) / sum(s_1 == s_u[0])
p[1, 0] = sum((s_1 == s_u[1]) & (s_2 == s_u[0])) / sum(s_1 == s_u[1])
p[1, 1] = sum((s_1 == s_u[1]) & (s_2 == s_u[1])) / sum(s_1 == s_u[1])
p = maximum(k, p)
p[0] = p[0] / sum(p[0])
p[1] = p[1] / sum(p[1])
