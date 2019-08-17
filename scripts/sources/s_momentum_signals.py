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

# # s_momentum_signals [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_momentum_signals&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-mom-signal).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from arpym.statistics import ewm_meancov
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_momentum_signals-parameters)

t_start = 252*2  # starting date in the plot
w_smooth = 180  # trailing window for smoothing
w_score = 252  # strailing window for scoring
tau_hl_smooth = 10  # half-life for smoothing
tau_hl_score = 120  # half-life for scoring
n_1 = 0.2  # index of first signal for comparison (will be round of n_*n_1)
n_2 = 0.4  # index of second signal for comparison (will be round of n_*n_2)
n_3 = 0.6  # index of third signal for comparison (will be round of n_*n_3)
n_4 = 0.8  # index of fourth signal for comparison (will be round of n_*n_4)

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_momentum_signals-implementation-step00): Load data

path = '../../../databases/global-databases/strategies/db_strategies/'
s_mom = pd.read_csv(path + 's_mom.csv', index_col=0, parse_dates=True)
v = pd.read_csv(path + 'last_price.csv', index_col=0, parse_dates=True)
dates = pd.to_datetime(s_mom.index).date
s_mom = np.array(s_mom)
v = np.array(v)
t_, n_ = s_mom.shape  # number of observations and number of stocks
n_1 = int(np.around(n_*n_1))
n_2 = int(np.around(n_*n_2))
n_3 = int(np.around(n_*n_3))
n_4 = int(np.around(n_*n_4))

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_momentum_signals-implementation-step01): Compute the smoothed signals

# +
s_mom_smoo = np.zeros((s_mom.shape[0] - w_smooth + 1, n_))

for t in range(w_smooth, s_mom.shape[0] + 1):
    s_mom_smoo[t - w_smooth, :] = ewm_meancov(s_mom[t - w_smooth:t, :],
                                               tau_hl_smooth)[0]
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_momentum_signals-implementation-step02): Compute the scored signals

# +
s_mom_scor = np.zeros((s_mom.shape[0] - w_score + 1, n_))
lambda_score = np.log(2) / tau_hl_score  # decay rate
p_scor = np.exp(-lambda_score*np.arange(w_score)[::-1]).reshape(-1) /\
         np.sum(np.exp(-lambda_score*np.arange(w_score)[::-1]))

for t in range(w_score, s_mom.shape[0] + 1):
    ewma, ewm_cov = ewm_meancov(s_mom_smoo[t - w_score:t, :], tau_hl_score)
    ewm_sd = np.sqrt(np.diag(ewm_cov))
    s_mom_scor[t - w_score, :] = (s_mom_smoo[t - w_smooth, :] - ewma) / ewm_sd


# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_momentum_signals-implementation-step03): Compute the ranked signals

s_mom_rk = np.zeros((s_mom_scor.shape[0], n_))

for t in range(s_mom_scor.shape[0]):
    rk = np.argsort(s_mom_scor[t, :])
    rk_signal = np.argsort(rk)
    s_mom_rk[t, :] = (rk_signal)*(2 / n_) - 1


# ## Save the data

output = {'t_': pd.Series(t_),
          'n_': pd.Series(n_),
          'w_smooth': pd.Series(w_smooth),
          'w_score': pd.Series(w_score),
          't_start': pd.Series(t_start),
          'dates': pd.Series(dates),
          'v': pd.Series(v[:, :n_].reshape((t_*n_,))),
          's_mom_rk': pd.Series(s_mom_rk.reshape(((t_-w_score+1)*n_,))),
          's_mom_scor': pd.Series(s_mom_scor.reshape(((t_-w_score+1)*n_,))),
          's_mom_smoo': pd.Series(s_mom_smoo.reshape(((t_-w_smooth+1)*n_,)))}

df = pd.DataFrame(output)
df.to_csv('../../../databases/temporary-databases/db_signals_mom.csv', index=None)
# -

# ## Plots

# +
# Compare the plots of one signal, one smoothed signal and one scored signal
plt.style.use('arpm')

t_start = t_start-1
dates = dates[t_start:]
grid_dates = np.linspace(0, len(dates)-1, 5)
grid_dates = list(map(int, np.around(grid_dates)))  # integer numbers

dates_dt = []
for i in dates:
    dates_dt.append(i)
dates_dt = np.array(dates_dt)
date_tick = grid_dates
myFmt = mdates.DateFormatter('%d-%b-%Y')
index = np.argsort(s_mom_rk[-1, :])

# Compare the plots of a cluster of 4 scored signals with their ranked
# counterparts
fig1, ax = plt.subplots(2, 1)
plt.sca(ax[0])
xx = t_start
plt.plot(dates_dt, s_mom[xx:, index[n_1]])
plt.xticks(dates_dt[date_tick])
ax[0].xaxis.set_major_formatter(myFmt)
plt.xlim([dates_dt[0], dates_dt[-1]])
plt.ylim([np.min(s_mom[xx:, index[n_1]])-.05*np.max(s_mom[xx:, index[n_1]]),
          np.max(s_mom[xx:, index[n_1]])+.05*np.max(s_mom[xx:, index[n_1]])])
plt.title('Momentum versus smoothed momentum signal')
xx = t_start - w_smooth + 1
plt.plot(dates_dt, s_mom_smoo[xx:, index[n_1]], 'r')
plt.xticks(dates_dt[date_tick])

plt.sca(ax[1])
xx = t_start - w_score + 1
plt.plot(dates_dt, s_mom_scor[xx:, index[n_1]])
plt.xticks(dates_dt[date_tick])
ax[1].xaxis.set_major_formatter(myFmt)
plt.xlim([dates_dt[0], dates_dt[-1]])
plt.title('Scored momentum signal')
add_logo(fig1, axis=ax[1], location=1)
plt.tight_layout()

fig2, ax = plt.subplots(2, 1)
plt.sca(ax[0])
plt.plot(dates_dt, s_mom_scor[xx:, [index[n_1], index[n_2],
                                    index[n_3], index[n_4]]])
plt.xticks(dates_dt[date_tick])
ax[0].xaxis.set_major_formatter(myFmt)
plt.xlim([dates_dt[0], dates_dt[-1]])
plt.title('Scored momentum signal cluster')

plt.sca(ax[1])
plt.plot(dates_dt, s_mom_rk[xx:, [index[n_1], index[n_2],
                                  index[n_3], index[n_4]]])
plt.xticks(dates_dt[date_tick])
ax[1].xaxis.set_major_formatter(myFmt)
plt.xlim([dates_dt[0], dates_dt[-1]])
plt.ylim([-1.05, 1.05])
plt.title('Ranked momentum signal cluster')
add_logo(fig2, axis=ax[0], location=4)
plt.tight_layout()
