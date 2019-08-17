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

# # s_high_freq_stock_var [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_high_freq_stock_var&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerMktMicroStructure).

# +
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from arpym.tools import trade_quote_processing, trade_quote_spreading, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_high_freq_stock_var-parameters)

i_0 = 0  # index of window's starting time for quotes
i_1 = 1249  # index of window's last time for quotes

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_high_freq_stock_var-implementation-step00): Load data

# +
path = '../../../databases/global-databases/high-frequency/\
db_US_10yr_Future_quotestrades/'
quotes = pd.read_csv(path + 'quotes.csv', index_col=0, parse_dates=True)
trades = pd.read_csv(path + 'trades.csv', index_col=0, parse_dates=True)

dates_quotes = pd.to_datetime(quotes.index).date
t0 = pd.to_datetime(quotes.index)
time_quotes = np.zeros(len(t0))  # time vector of quotes
for i in range(len(time_quotes)):
    time_quotes[i] = t0[i].timestamp()
p_bid = np.array(quotes.loc[:, 'bid'])  # best bids
p_ask = np.array(quotes.loc[:, 'ask'])  # best asks
h_bid = np.array(quotes.loc[:, 'bsiz'])  # bid sizes
h_ask = np.array(quotes.loc[:, 'asiz'])  # ask sizes

dates_trades = pd.to_datetime(trades.index).date
t_k0 = pd.to_datetime(trades.index)
time_trades = np.zeros(len(t_k0))  # time vector of trades
for i in range(len(time_trades)):
    time_trades[i] = t_k0[i].timestamp()
p_last = np.array(trades.loc[:, 'price'])  # last transaction values
delta_q = np.array(trades.loc[:, 'siz'])  # flow of traded contracts' sizes
delta_sgn = np.array(trades.loc[:, 'aggress'])  # trade sign flow
match = np.array(trades.loc[:, 'mtch'])  # match events
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_high_freq_stock_var-implementation-step01): Process the time series

# +
# process data
time_quotes, _, h_ask, p_ask, h_bid, p_bid, time_trades, _, p_last, delta_q,\
        delta_sgn, _ = trade_quote_processing(time_quotes, dates_quotes, h_ask,
                                              p_ask, h_bid, p_bid, time_trades,
                                              dates_trades, p_last, delta_q,
                                              delta_sgn, match)

time_quotes = time_quotes.flatten()

# index of the first trade within the time window
k_0 = np.where(time_trades >= time_quotes[i_0])[0][0]
# index of the last trade within the time window
k_1 = np.where(time_trades <= time_quotes[i_1])[0][-1]
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_high_freq_stock_var-implementation-step02): Compute the market microstructure variables

# +
q = np.cumsum(delta_q)  # cumulative volume series
sgn = np.cumsum(delta_sgn)  # cumulative trade sign series

# number of millisecond points in the time window
ms = int(np.around((time_quotes[i_1]-time_quotes[i_0])*1000))+1
# spreading wall-clock-time vector in milliseconds
t_ms = np.linspace(time_quotes[i_0], time_quotes[i_1], int(ms))

# spreading time series
h_ask, p_ask, h_bid, p_bid, _, _, _, \
    _ = trade_quote_spreading(t_ms, time_quotes[i_0:i_1], h_ask[i_0:i_1],
                              p_ask[i_0:i_1], h_bid[i_0:i_1],
                              p_bid[i_0:i_1], time_trades[k_0:k_1],
                              p_last[k_0:k_1], q[k_0:k_1], sgn[k_0:k_1])
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_high_freq_stock_var-implementation-step03): Compute p_mic and histograms of the bid and ask sizes

p_mic = (p_bid * h_ask + p_ask * h_bid) / (h_ask + h_bid)  # microprice series

# ## Plots

# +
# rescale h_bid and h_ask
h_bid_res = p_bid - h_bid / 100000
h_ask_res = p_ask + h_ask / 100000

plt.style.use('arpm')

# axes settings
ymax_1 = np.max(h_ask_res) + 0.02
ymin_1 = np.min(h_bid_res) - 0.01
ytick_1 = np.linspace(ymin_1, ymax_1, 5)
t_ms_dt = []
for i in t_ms:
    t_ms_dt.append(datetime.fromtimestamp(i))
t_ms_dt = np.array(t_ms_dt)

trans_time = []
for i in range(k_0, k_1):
    trans_time.append(datetime.fromtimestamp(time_trades[i]))
trans_time = np.array(trans_time)

fig = plt.figure()

plt.subplot(211)
plt.axis([min(t_ms_dt), max(t_ms_dt), ymin_1, ymax_1])
plt.yticks(ytick_1)
plt.ylabel('price')
plt.xlabel('time')

plt.title('TAQ data for US 10yr Future: {date}'.
          format(date=t_ms_dt[0].strftime('%Y-%b-%d')))
plt.grid(True)

for k in range(k_0, k_1-1):
    plt.plot([trans_time[k-k_0], trans_time[k-k_0+1]],
             [p_last[k], p_last[k]], lw=1, color='b')

plt.scatter(trans_time, p_last[range(k_0, k_1)], c='b', s=20, label='traded')

plt.plot(t_ms_dt, h_bid_res, color=[.8, .8, .8], lw=1.2)
plt.plot(t_ms_dt, p_mic, color='r', lw=1.4, label='micro')
plt.plot(t_ms_dt, p_bid, color=[0, .6, 0], lw=1.4, label='bid and ask')
plt.plot(t_ms_dt, h_ask_res, color=[.8, .8, .8], lw=1.2,
         label='bid and ask size')
plt.plot(t_ms_dt, p_ask, color=[0, .6, 0], lw=1.4)
plt.legend()

dt = 100
for i in range(2 * dt, len(t_ms_dt) - dt, dt):
    plt.plot([t_ms_dt[i], t_ms_dt[i]], [h_bid_res[i], p_bid[i] - 0.0007],
             color=[.8, .8, .8], linestyle='-')
    plt.plot([t_ms_dt[i], t_ms_dt[i]], [p_ask[i] + 0.0007, h_ask_res[i]],
             color=[.8, .8, .8], linestyle='-')

ax1 = plt.subplot(212)
ax1.yaxis.label.set_color('red')
ax1.set_ylabel('Cumulative volume')
ax1.set_xlabel('time')
ymax_2 = np.max(q[range(k_0, k_1)]) + 30
ymin_2 = np.min(q[range(k_0, k_1)])
ax1.set_xlim(min(t_ms_dt), max(t_ms_dt))
ax1.set_ylim(ymin_2, ymax_2)
ax1.step(trans_time, q[range(k_0, k_1)], color='r', where='post')
ax1.plot(trans_time, q[range(k_0, k_1)], '.', color='r', markersize=10)

ax2 = ax1.twinx()
ax2.yaxis.label.set_color('green')
ax2.set_ylabel("Cumulative sign")
ymax_3 = np.max(sgn[range(k_0, k_1)]) + 1
ymin_3 = np.min(sgn[range(k_0, k_1)])
ax2.set_ylim(ymin_3, ymax_3)
ax2.set_xlim(min(t_ms_dt), max(t_ms_dt))
ax2.step(trans_time, sgn[range(k_0, k_1)], color='g', where='post')
ax2.plot(trans_time, sgn[range(k_0, k_1)], '.', color='g', markersize=10)
add_logo(fig, location=5)
plt.tight_layout()
