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

# # s_cointegration_signal [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_cointegration_signal&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-signals-cointegration).

# +
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from arpym.estimation import cointegration_fp, fit_var1, var2mvou
from arpym.tools import trade_quote_processing, trade_quote_spreading, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_cointegration_signal-parameters)

delta_a = 10000  # time binning

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_cointegration_signal-implementation-step00): Load data

# +
path = '../../../databases/global-databases/high-frequency/db_stocks_highfreq/'

AMZN_q = pd.read_csv(path + 'AMZN/quote.csv', index_col=0, parse_dates=True)
AMZN_t = pd.read_csv(path + 'AMZN/trade.csv', index_col=0, parse_dates=True)

GOOG_q = pd.read_csv(path + 'GOOG/quote.csv', index_col=0, parse_dates=True)
GOOG_t = pd.read_csv(path + 'GOOG/trade.csv', index_col=0, parse_dates=True)

# Amazon quotes
t_A = np.array([pd.to_datetime(AMZN_q.index)[i].timestamp() for i
                in range(len(AMZN_q.index))])
dates_quotes_A = np.array(pd.to_datetime(AMZN_q.index).date)
q_ask_A = AMZN_q['asksize'].values
p_ask_A = AMZN_q['ask'].values
q_bid_A = AMZN_q['bidsize'].values
p_bid_A = AMZN_q['bid'].values
# Amazon trades
t_q_A = np.array([pd.to_datetime(AMZN_t.index)[i].timestamp() for i
                  in range(len(AMZN_t.index))])
dates_trades_A = np.array(pd.to_datetime(AMZN_t.index).date)
p_last_A = AMZN_t['price'].values
delta_q_A = AMZN_t['volume'].values
delta_sgn_A = AMZN_t['sign'].values
match_A = AMZN_t['match'].values

# Google quotes
t_G = np.array([pd.to_datetime(GOOG_q.index)[i].timestamp() for i
                in range(len(GOOG_q.index))])
dates_quotes_G = np.array(pd.to_datetime(GOOG_q.index).date)
q_ask_G = GOOG_q['asksize'].values
p_ask_G = GOOG_q['ask'].values
q_bid_G = GOOG_q['bidsize'].values
p_bid_G = GOOG_q['bid'].values
# Google trades
t_G_q = np.array([pd.to_datetime(GOOG_t.index)[i].timestamp() for i
                  in range(len(GOOG_t.index))])
dates_trades_G = np.array(pd.to_datetime(GOOG_t.index).date)
p_last_G = GOOG_t['price'].values
delta_q_G = GOOG_t['volume'].values
delta_sgn_G = GOOG_t['sign'].values
match_G = GOOG_t['match'].values
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_cointegration_signal-implementation-step01): Process the database and compute cumulative volume for each stock

# +
t_A_p, dates_quotes_A_p, q_ask_A_p, p_ask_A_p, q_bid_A_p, p_bid_A_p, t_A_n_p,\
 dates_trades_A_p, p_last_A_p, delta_q_A_p, delta_sgn_A_p, _ = \
 trade_quote_processing(t_A, dates_quotes_A, q_ask_A, p_ask_A, q_bid_A,
                        p_bid_A, t_q_A, dates_trades_A, p_last_A, delta_q_A,
                        delta_sgn_A, match_A)

t_A_p = t_A_p.flatten()
t_A_n_p = t_A_n_p.flatten()

t_G_p, dates_quotes_G_p, q_ask_G_p, p_ask_G_p, q_bid_G_p, p_bid_G_p, t_G_n_p,\
 dates_trades_G_p, p_last_G_p, delta_q_G_p, delta_sgn_G_p, _ = \
 trade_quote_processing(t_G, dates_quotes_G, q_ask_G, p_ask_G, q_bid_G,
                        p_bid_G, t_G_q, dates_trades_G, p_last_G, delta_q_G,
                        delta_sgn_G, match_G)

t_G_p = t_G_p.flatten()
t_G_n_p = t_G_n_p.flatten()

q_A_t = np.cumsum(delta_q_A_p)  # Amazon cumulative volume
q_G_t = np.cumsum(delta_q_G_p)  # Google cumulative volume
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_cointegration_signal-implementation-step02): Compute ask/bid sizes, prices, and cumulative volumes

# +
time_vec_A = datetime.fromtimestamp(t_A_n_p[-1]) - \
 datetime.fromtimestamp(t_A_n_p[0])
# total length of time window expressed in wall-clock-time for Amazon stock
ms_A = time_vec_A.seconds * 1000 + time_vec_A.microseconds / 1000
# time window's wall-clock-time vector expressed in milliseconds, Amazon stock
time_ms_A = np.linspace(t_A_n_p[0], t_A_n_p[-1], int(ms_A + 1))

h_ask_A_t, p_ask_A_t, h_bid_A_t, p_bid_A_t, _, q_A_t, *_\
 = trade_quote_spreading(time_ms_A, t_A_p, q_ask_A_p, p_ask_A_p,
                         q_bid_A_p, p_bid_A_p, t_A_n_p, p_last_A_p,
                         q_A_t, delta_sgn_A_p)

time_vec_G = datetime.fromtimestamp(t_G_n_p[-1]) - \
 datetime.fromtimestamp(t_G_n_p[0])
# total length of time window expressed in wall-clock-time for Google stock
ms_G = time_vec_G.seconds * 1000 + time_vec_G.microseconds / 1000
# time window's wall-clock-time vector expressed in milliseconds. Google stock
time_ms_G = np.linspace(t_G_n_p[0], t_G_n_p[-1], int(ms_G+1))

h_ask_G_t, p_ask_G_t, h_bid_G_t, p_bid_G_t, _, q_G_t, *_\
 = trade_quote_spreading(time_ms_G, t_G_p,
                         q_ask_G_p, p_ask_G_p,
                         q_bid_G_p, p_bid_G_p,
                         t_G_n_p, p_last_G_p,
                         q_G_t, delta_sgn_G_p)

q_A_t = q_A_t.flatten()
q_G_t = q_G_t.flatten()
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_cointegration_signal-implementation-step03): Compute ask/bid sizes, prices in common wall clock-time, microprices, and common activity time

# +
time_ms = np.intersect1d(time_ms_A, time_ms_G)
i_A = np.argwhere(np.in1d(time_ms_A, time_ms) == 1).flatten()
i_G = np.argwhere(np.in1d(time_ms_G, time_ms) == 1).flatten()

h_ask_A_ct = h_ask_A_t[i_A]
p_ask_A_ct = p_ask_A_t[i_A]
h_bid_A_ct = h_bid_A_t[i_A]
p_bid_A_ct = p_bid_A_t[i_A]

h_ask_G_ct = h_ask_G_t[i_G]
p_ask_G_ct = p_ask_G_t[i_G]
h_bid_G_ct = h_bid_G_t[i_G]
p_bid_G_ct = p_bid_G_t[i_G]

# clock time microprice series
p_mic_A_t = (p_bid_A_ct * h_ask_A_ct + p_ask_A_ct * h_bid_A_ct) /\
 (h_ask_A_ct + h_bid_A_ct)

p_mic_G_t = (p_bid_G_ct * h_ask_G_ct + p_ask_G_ct * h_bid_G_ct) /\
 (h_ask_G_ct + h_bid_G_ct)

# substitute the zeros entries in the cumulative volumes with the last nonzeros
for i in np.where(q_A_t == 0)[0]:
    q_A_t[i] = q_A_t[i - 1]

for i in np.where(q_G_t == 0)[0]:
    q_G_t[i] = q_G_t[i - 1]

# cumulative volumes in common wall clock time
q_A_t_c = q_A_t[i_A]
q_G_t_c = q_G_t[i_G]

sum_vol = q_A_t_c * p_mic_A_t + q_G_t_c * p_mic_G_t

delta_a = 10000  # width of activity time bins
amin = np.min(sum_vol)
amax = np.max(sum_vol)
a_t = np.arange(amin, amax + delta_a, delta_a)  # common activity time
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_cointegration_signal-implementation-step04): Compute two stocks microprice series in common activity time

# +
sum_vol_asc, indices = np.unique(sum_vol, return_index=True)

p_mic_A_at = np.interp(a_t, sum_vol_asc, p_mic_A_t[indices])
p_mic_G_at = np.interp(a_t, sum_vol_asc, p_mic_G_t[indices])
# -

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_cointegration_signal-implementation-step05): Calculate cointegrated vectors, cointegrated series y_t, fit an AR(1) model to the series y_t

# +
x_t = np.r_['0,2', p_mic_A_at, p_mic_G_at].T
c_hat, *_ = cointegration_fp(x_t)  # cointegrated vectors
c = c_hat[:, 1]

# cointegrated series corresponding to the second cointegration vector
y_t = x_t @ c

b_hat, mu_hat_epsi, sig2_hat_epsi = fit_var1(y_t)
dt = p_mic_A_at.shape[0] / 100  # time steps
theta, mu, sigma2 = var2mvou(b_hat, mu_hat_epsi, sig2_hat_epsi, dt)

mu_infty = np.linalg.solve(theta, mu)  # long-run expectation
sigma_infty = np.sqrt(sigma2 / (2 * theta))  # long-run standard deviation
# -

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_cointegration_signal-implementation-step06): Compute clock time corresponding to the cointegrated series in common activity time

t = np.interp(a_t, sum_vol_asc, time_ms[indices])

# ## Plots

# +
plt.style.use('arpm')

# color settings
orange = [.9, .3, .0]
blue = [0, 0, .8]
xtick = np.linspace(a_t[0], a_t[-1], 5)

fig, ax = plt.subplots(2, 1)

# microprice series in common volume-activity time
plt.sca(ax[0])
plt.title('Microprice series in common volume-activity time')
plt.plot(a_t, p_mic_A_at, color=orange)
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.5e'))
plt.xticks(xtick)
plt.axis([amin, amax, np.min(p_mic_A_at), np.max(p_mic_A_at)])
plt.yticks(np.linspace(np.min(p_mic_A_at), np.max(p_mic_A_at), 5))
plt.xlabel('Common volume-activity time')
plt.ylabel('Amazon microprice', color=orange)
ax2 = ax[0].twinx()
plt.plot(a_t, p_mic_G_at, color=blue)
plt.axis([amin, amax, np.min(p_mic_G_at), np.max(p_mic_G_at)])
plt.yticks(np.linspace(np.min(p_mic_G_at), np.max(p_mic_G_at), 5))
plt.ylabel('Google microprice', color=blue)
plt.grid(True)

# cointegrated series in common activity time
plt.sca(ax[1])
plt.title('Cointegrated microprice series in common volume-activity time')
plt.plot(a_t, y_t, color='k')
plt.plot([amin, amax], np.tile(mu_infty, 2), label='Mean', color='g')
plt. plot([amin, amax], np.tile(mu_infty + 2 * sigma_infty.squeeze(), 2),
          label=' + / - 2 Std. deviation', color='r')
plt.plot([amin, amax], np.tile(mu_infty - 2 * sigma_infty.squeeze(), 2),
         color='r')
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.5e'))
plt.xticks(xtick)
plt.axis([amin, amax, np.min(y_t), np.max(y_t)])
plt.yticks(np.linspace(np.min(y_t), np.max(y_t), 5))
plt.xlabel('Common volume-activity time')
plt.ylabel('Cointegrated series')
plt.legend(loc=4)
plt.grid(True)
pos1 = ax[1].get_position()

add_logo(fig, location=8)
plt.tight_layout()

# cointegrated series in clock time
fig, ax = plt.subplots(figsize=(10, 4))
plt.sca(ax)
time_dt = np.array([datetime.fromtimestamp(a) for a in t])
plt.title('Cointegrated microprice series in clock time')
plt.plot(time_dt, y_t, color='k')
plt.plot(time_dt, np.tile(mu_infty, len(t)),
         label='Mean', color='g')
plt.plot(time_dt, np.tile(mu_infty + 2*sigma_infty.squeeze(),
                          len(t)),
         label=' + / - 2 Std. deviation', color='r')
plt.plot(time_dt, np.tile(mu_infty - 2*sigma_infty.squeeze(),
                          len(t)), color='r')
plt.axis([np.min(time_dt), np.max(time_dt), np.min(y_t), np.max(y_t)])
plt.yticks(np.linspace(np.min(y_t), np.max(y_t), 5))
plt.xlabel('Clock time')
plt.ylabel('Cointegrated series')
plt.legend(loc=4)
plt.grid(True)
plt.subplots_adjust(right=pos1.x0 + pos1.width)
add_logo(fig, set_fig_size=False, location=8)
plt.tight_layout()
