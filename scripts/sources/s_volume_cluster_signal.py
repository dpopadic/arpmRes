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

# # s_volume_cluster_signal [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_volume_cluster_signal&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-signals-volume-clustering).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from arpym.tools import trade_quote_processing, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_volume_cluster_signal-parameters)

k_0 = 121  # index of the first trade within the time window
k_1 = 210  # index of the last trade within the time window
tau_hl = 5  # decay rate
w = 30  # trailing window

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_volume_cluster_signal-implementation-step00): Load data

# +
path = '../../../databases/global-databases/high-frequency/db_US_10yr_Future_quotestrades/'
quotes = pd.read_csv(path + 'quotes.csv', index_col=0, parse_dates=True)
trades = pd.read_csv(path + 'trades.csv', index_col=0, parse_dates=True)

dates_quotes = pd.to_datetime(quotes.index).date
# time vector of quotes
t = np.array(list(map(lambda x: x.timestamp(), pd.to_datetime(quotes.index))))
p_bid = np.array(quotes.loc[:, 'bid'])  # best bids
p_ask = np.array(quotes.loc[:, 'ask'])  # best asks
q_bid = np.array(quotes.loc[:, 'bsiz'])  # bid sizes
q_ask = np.array(quotes.loc[:, 'asiz'])  # ask sizes

dates_trades = pd.to_datetime(trades.index).date
# time vector of trades
t_k = np.array(list(map(lambda x: x.timestamp(),
                        pd.to_datetime(trades.index))))
p_last = np.array(trades.loc[:, 'price'])  # last transaction values
delta_q = np.array(trades.loc[:, 'siz'])  # flow of traded contracts' sizes
delta_sgn = np.array(trades.loc[:, 'aggress'])  # trade sign flow
match = np.array(trades.loc[:, 'mtch'])  # match events
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_volume_cluster_signal-implementation-step01): Process the database

t, _, q_ask, p_ask, q_bid, p_bid, t_k, _, p_last, delta_q, _,\
       _ = trade_quote_processing(t, dates_quotes, q_ask, p_ask, q_bid,
                                  p_bid, t_k, dates_trades, p_last, delta_q,
                                  delta_sgn, match)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_volume_cluster_signal-implementation-step02): Compute the traded price, the bid/ask prices, the bid/ask sizes and the microprice

# +
tick_time = np.arange(len(p_last[k_0:k_1+1]))
i_ = len(tick_time)
# last transaction value within the time window as a function of tick time
p_last_k = p_last[k_0:k_1+1]  # traded price

# indexes of bid/ask prices near to the traded prices
ti = np.zeros(i_, dtype=int)
for i in range(i_):
    ti[i] = np.where(t <= t_k[k_0+i])[0][-1]

p_ask = p_ask[ti]  # ask price in tick time
p_bid = p_bid[ti]  # bid price in tick time
q_bid = q_bid[ti]
q_ask = q_ask[ti]
# microprice in tick time
p_mic = (p_bid * q_ask+p_ask * q_bid) / (q_ask+q_bid)
p_mid = (p_bid + p_ask) / 2  # mid-price in tick time
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_volume_cluster_signal-implementation-step03): Compute the volume clustering signal

# +
s_vol_clus = np.zeros((i_,))  # initialization
nu = np.log(2) / tau_hl
gamma_w = 1 + sum(np.exp(-nu*np.arange(1, w,)))
s_vol_clus[0] = 1 / gamma_w*(delta_q[k_0] +
                             sum(np.exp((-nu) * np.arange(1, w,)) *
                                 delta_q[k_0:k_0-(w - 1):-1]))

for i in range(i_):
    s_vol_clus[i] = (1 - np.exp(-nu)) *\
                        delta_q[k_0 + i] +\
                        np.exp(-nu) * s_vol_clus[i-1]
# -

# ## Plots

# +
plt.style.use('arpm')

# colors
lgray = [0.8, 0.8, 0.8]
dgreen = [0, 0.6, 0]
orange = [0.94, 0.35, 0]
dred = [0.8, 0, 0.2]
t_dt = []
for i in t:
    t_dt.append(datetime.fromtimestamp(i))
t_dt = np.array(t_dt)

# microprice, bid/ask price, bid/ask size, transaction value, mid-price
fig = plt.figure()
plt.subplot2grid((2, 1), (0, 0))  # axes settings
q_bid_res = p_bid-q_bid / 100000  # q_bid rescaled
q_ask_res = p_ask+q_ask / 100000  # q_ask rescaled
xtick = np.linspace(tick_time[0], tick_time[-1], 7, dtype=int)

plt.axis([np.min(tick_time), np.max(tick_time), 132.41, 132.53])
plt.xticks(xtick)
plt.yticks(np.arange(132.36, 132.53 + 0.02, 0.02))
plt.ylabel('Price')
plt.title('US 10 yr Future: {date}'.format(date=t_dt[0].strftime('%Y-%b-%d')))
plt.grid(True)
plt.plot(tick_time, q_bid_res, color=lgray)
p0 = plt.plot(tick_time, q_ask_res, color=lgray, label='bid/ask size')
p1 = plt.plot(tick_time, p_bid, color=dgreen, label='bid/ask price')
plt.plot(tick_time, p_ask, color=dgreen)
p3 = plt.plot([tick_time[:i_], tick_time[:i_]],
              [p_last[k_0:k_1+1], p_last[k_0:k_1+1]], c='b', marker='.',
              label='traded price')
p2 = plt.plot(tick_time, p_mic, color=orange, label='microprice')
plt.legend(handles=[p0[0], p1[0], p2[0], p3[0]])

# signal: exponential moving average of the traded volume with a fast decay
plt.subplot2grid((2, 1), (1, 0))
plt.axis([min(tick_time), max(tick_time), 0, 155])
plt.xticks(xtick)
plt.yticks(np.arange(0, 200, 50))

p4 = plt.plot(tick_time, delta_q[k_0:k_1+1], color='c', marker='.',
              label='traded volume')
maxticktime = len(tick_time) - 1
p5 = plt.plot([tick_time[:maxticktime], tick_time[:maxticktime]],
              [s_vol_clus[:maxticktime], s_vol_clus[:maxticktime]],
              lw=1, color='k', marker='.', label='signal')

p6 = plt.plot(tick_time, np.tile(30, i_), color=dred,
              label='increase order trigger')

plt.legend(handles=[p4[0], p5[0], p6[0]])
plt.ylabel('Volume')
plt.xlabel('Tick time')
plt.title('Signal: exponential moving average of the traded volume')
add_logo(fig, location=6)
plt.tight_layout()
