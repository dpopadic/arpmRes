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

# # s_high_freq_tick_time [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_high_freq_tick_time&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerTickTEvol).

# +
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from arpym.tools import trade_quote_processing, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_high_freq_tick_time-parameters)

i_0 = 0  # index of window's starting time for quotes
i_1 = 1249  # index of window's last time for quotes

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_high_freq_tick_time-implementation-step00): Load data

# +
path = '../../../databases/global-databases/high-frequency/\
db_US_10yr_Future_quotestrades/'
quotes = pd.read_csv(path + 'quotes.csv', index_col=0, parse_dates=True)
trades = pd.read_csv(path + 'trades.csv', index_col=0, parse_dates=True)

dates_quotes = pd.to_datetime(quotes.index).date
t0 = pd.to_datetime(quotes.index)

time_quotes = np.zeros(len(t0))
for i in range(len(time_quotes)):
    time_quotes[i] = t0[i].timestamp()
p_bid = np.array(quotes.loc[:, 'bid'])  # best bids
p_ask = np.array(quotes.loc[:, 'ask'])  # best asks
h_bid = np.array(quotes.loc[:, 'bsiz'])  # bid sizes
h_ask = np.array(quotes.loc[:, 'asiz'])  # ask sizes

dates_trades = pd.to_datetime(trades.index).date
t_k0 = pd.to_datetime(trades.index)  # time vector of trades
time_trades = np.zeros(len(t_k0))
for i in range(len(time_trades)):
    time_trades[i] = t_k0[i].timestamp()
p_last = np.array(trades.loc[:, 'price'])  # last transaction values
delta_q = np.array(trades.loc[:, 'siz'])  # flow of traded contracts' sizes
delta_sgn = np.array(trades.loc[:, 'aggress'])  # trade sign flow
match = np.array(trades.loc[:, 'mtch'])  # match events
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_high_freq_tick_time-implementation-step01): Process the time series

# +
time_quotes, _, _, _, _, _, time_trades, _, p_last,\
    _, _, _ = trade_quote_processing(time_quotes, dates_quotes,
                                     h_ask, p_ask, h_bid,
                                     p_bid, time_trades, dates_trades, p_last,
                                     delta_q, delta_sgn, match)

time_quotes = time_quotes.flatten()

# index of the first trade within the time window
k_0 = np.where(time_trades >= time_quotes[i_0])[0][0]
# index of the last trade within the time window
k_1 = np.where(time_trades <= time_quotes[i_1])[0][-1]

# last transaction value within the time window as a function of tick time
p_last_k = p_last[k_0: k_1+1]
# number of trades within the time window as a function of tick time
k_t = np.arange(1, len(p_last_k)+1)
# -

# ## Plots

# +
plt.style.use('arpm')
trans_time = []
for i in range(k_0, k_1+1):
    trans_time.append(datetime.fromtimestamp(time_trades[i]))
trans_time = np.array(trans_time)

# axes settings
pgrid_min = min(p_last_k)-0.001
pgrid_max = max(p_last_k)+0.001
pgrid = np.linspace(pgrid_min, pgrid_max, 5)
kgrid_min = min(k_t)
kgrid_max = max(k_t)+1
kgrid = np.linspace(kgrid_min, kgrid_max, 5, dtype=int)

fig, _ = plt.subplots(2, 2)

# top-left plot
plt.subplot(221)
plt.ticklabel_format(useOffset=False)
plt.yticks(pgrid)
plt.axis([min(trans_time), max(trans_time), pgrid_min, pgrid_max])
plt.ylabel('Transaction Price')
plt.xlabel('Wall Clock Time')
plt.title('Time evolution')

for k in range(len(k_t) - 1):
    plt.plot([trans_time[k], trans_time[k+1]],
             [p_last_k[k], p_last_k[k]], lw=1, color='r')
plt.scatter(trans_time, p_last_k, c='r', s=5)

plt.grid(True)

# top-right plot
plt.subplot(222)
plt.ticklabel_format(useOffset=False)
plt.xticks(kgrid)
plt.yticks(pgrid)
plt.axis([kgrid_min, kgrid_max, pgrid_min, pgrid_max])
plt.ylabel('Transaction Price')
plt.xlabel('Tick Time')
plt.title('Tick Time Activity Evolution')

for k in range(len(k_t) - 1):
    plt.plot([k_t[k], k_t[k+1]], [p_last_k[k], p_last_k[k]], lw=1, color='r')

plt.scatter(k_t, p_last_k, marker='.', s=5, color='r')
plt.grid(True)

# bottom-left plot
plt.subplot(223)
plt.yticks(kgrid)
plt.axis([min(trans_time), max(trans_time), kgrid_min, kgrid_max])
plt.ylabel('Number of trades')
plt.xlabel('Wall Clock Time')
for k in range(len(k_t) - 1):
    plt.plot([trans_time[k], trans_time[k+1]], [k_t[k], k_t[k]], lw=1,
             color='b')
plt.scatter(trans_time, k_t, marker='.', s=5, color='b')
plt.grid(True)

# bottom-right plot
plt.subplot(224)
plt.yticks(kgrid)
plt.axis([kgrid_min, kgrid_max, kgrid_min, kgrid_max])
plt.ylabel('Number of trades')
plt.xlabel('Tick Time')
plt.plot(k_t, k_t, lw=1, color='b')
plt.grid(True)
add_logo(fig, size_frac_x=1/8)
plt.tight_layout()
