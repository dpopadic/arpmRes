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

# # s_price_pred_signal [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_price_pred_signal&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-signals-mark-to-market-value).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from arpym.tools import trade_quote_processing, add_logo
from arpym.statistics import ewm_meancov
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_price_pred_signal-parameters)

k_0 = 208  # index of the first trade within the time window
k_1 = 404  # index of the last trade within the time window
tau_hl = 5  # decay rate
w = 10  # trailing window

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_price_pred_signal-implementation-step00): Load data

# +
path = '../../../databases/global-databases/high-frequency/' + \
    'db_US_10yr_Future_quotestrades/'
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

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_price_pred_signal-implementation-step01): Process the database

t, _, q_ask, p_ask, q_bid, p_bid, t_k, _, p_last, delta_q, _,\
       _ = trade_quote_processing(t, dates_quotes, q_ask, p_ask, q_bid,
                                  p_bid, t_k, dates_trades, p_last, delta_q,
                                  delta_sgn, match)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_price_pred_signal-implementation-step02): Compute the traded price, the bid/ask prices, the bid/ask sizes and the microprice

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
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_price_pred_signal-implementation-step03): Compute the decay rate, the price perdiction signal and the mid-price

# +
average_price = np.zeros((i_,))
ewma1 = np.zeros(i_)
ewma2 = np.zeros(i_)
dollar_volume = 0
volume = 0

for i in range(i_):
    ewma1[i] = ewm_meancov(
            p_last[k_0-w+i+1:k_0+i+1]*delta_q[k_0-w+i+1:k_0+i+1], tau_hl)[0]
    ewma2[i] = ewm_meancov(delta_q[k_0-w+i+1:k_0+i+1], tau_hl)[0]
    dollar_volume = dollar_volume + p_last_k[i] * delta_q[k_0:k_1+1][i]
    volume = volume + delta_q[k_0:k_1+1][i]
    average_price[i] = dollar_volume / volume

s_price_pred = ewma1 / ewma2
threshold = np.mean(average_price)
# -

# ## Plots

# +
plt.style.use('arpm')

# colors
lgray = [0.8, 0.8, 0.8]
orange = [0.93, 0.4, 0]
q_bid_res = p_bid-q_bid / 450000  # q_bid rescaled
q_ask_res = p_ask+q_ask / 450000  # q_ask rescaled

# axes settings
xtick = np.linspace(tick_time[0], tick_time[-1], 8, dtype=int)
ymax_1 = np.max(q_ask_res) + 0.001
ymin_1 = np.min(q_bid_res) - 0.001
ytick_1 = np.linspace(ymin_1, ymax_1, 5)

fig = plt.figure()

plt.axis([np.min(tick_time), np.max(tick_time), ymin_1, ymax_1])
plt.xticks(xtick)
plt.yticks(ytick_1)
plt.plot(tick_time, q_bid_res, color=lgray)
p0 = plt.plot(tick_time, q_ask_res, color=lgray,
              label='bid/ask price and size')
p2 = plt.plot(tick_time, p_mic, color=orange, label='microprice')
p5 = plt.plot(tick_time, average_price, color='c', label='average price')

for i in range(i_):
    plt.plot([tick_time[i], tick_time[i]], [q_bid_res[i], p_bid[i]],
             color=lgray, lw=3)
    plt.plot([tick_time[i], tick_time[i]], [p_ask[i], q_ask_res[i]],
             color=lgray, lw=3)
p3 = plt.plot([tick_time[:i_], tick_time[:i_]],
              [p_last_k[:i_], p_last_k[:i_]], markersize=3, color='b',
              marker='.', label='traded price')
p4 = plt.plot([tick_time[:i_], tick_time[:i_]],
              [s_price_pred[:i_], s_price_pred[:i_]], markersize=3, color='k',
              marker='.', label='signal')

plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

plt.legend(handles=[p0[0], p5[0], p2[0], p3[0], p4[0]])

plt.ylabel('Price')
plt.xlabel('Tick time')
plt.title('Mark-to-market price signal for US 10yr Future')
plt.grid(True)

add_logo(fig, location=9)
plt.tight_layout()
