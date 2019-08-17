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

# # s_stock_selection [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_stock_selection&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_stock_selection).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arpym.portfolio import obj_tracking_err
from arpym.statistics import meancov_sp
from arpym.estimation import exp_decay_fp
from arpym.tools import backward_selection, forward_selection, \
                        naive_selection, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_stock_selection-parameters)

n_ = 48  # number of stocks
k_ = 48   # number of selections
t_ = 1008  # length of the time series
t_now = '2012-01-01'  # current time
tau_hl = 180  # half life parameter

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_stock_selection-implementation-step00): Upload data

# +
path = '../../../databases/global-databases/equities/db_stocks_SP500/'
spx = pd.read_csv(path + 'SPX.csv', index_col=0, parse_dates=['date'])
stocks = pd.read_csv(path + 'db_stocks_sp.csv', skiprows=[0], index_col=0)
# merging datasets
spx_stocks = pd.merge(spx, stocks, left_index=True, right_index=True)
# select data within the date range
spx_stocks = spx_stocks.loc[spx_stocks.index <= t_now].tail(t_)
# remove the stocks with missing values
spx_stocks = spx_stocks.dropna(axis=1, how='any')
date = spx_stocks.index
# upload stocks values
v_stock = np.array(spx_stocks.iloc[:, 2:2+n_])

# upload S&P500 index value
v_sandp = np.array(spx_stocks.SPX_close)
t_ = v_stock.shape[0]
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_stock_selection-implementation-step01): Compute linear returns of both benchmark and securities

# +
# stocks return
r_stock = np.diff(v_stock, axis=0)/v_stock[:-1, :]

# S&P500 index return
r_sandp = np.diff(v_sandp, axis=0)/v_sandp[:-1]
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_stock_selection-implementation-step02): Cov. matrix of the joint vector of stocks and bench. returns

# +
# exponential decay probabilities
p = exp_decay_fp(t_ - 1, tau_hl)

# HFP covariance
_, s2_r_stock_r_sandp = meancov_sp(np.concatenate((r_stock, r_sandp.reshape(-1, 1)), axis=1), p)
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_stock_selection-implementation-step03): Objective function

g = lambda s: obj_tracking_err(s2_r_stock_r_sandp, s)[1]

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_stock_selection-implementation-step04): Portfolio selection via naive routine

s_star_naive = naive_selection(g, n_, k_)
g_te_naive = np.zeros(k_)
for k in np.arange(0, k_):
    g_te_naive[k] = g(s_star_naive[k])

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_stock_selection-implementation-step05): Portfolio selection via forward stepwise routine

s_star_fwd = forward_selection(g, n_, k_)
g_te_fwd = np.zeros(k_)
for k in np.arange(0, k_):
    g_te_fwd[k] = g(s_star_fwd[k])

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_stock_selection-implementation-step06): Portfolio selection via backward stepwise routine

s_star_bwd = backward_selection(g, n_, k_)
g_te_bwd = np.zeros(k_)
for k in np.arange(0, k_):
    g_te_bwd[k] = g(s_star_bwd[k])

# ## Plots

plt.style.use('arpm')
fig = plt.figure()
h3 = plt.plot(np.arange(1, k_+1), g_te_naive, color=[.5, .5, .5], lw=2,
              label='naive')
h1 = plt.plot(np.arange(1, k_ + 1), g_te_fwd, 'b',
              lw=2, label='forward stepwise')
h2 = plt.plot(np.arange(1, k_ + 1), g_te_bwd,
              color=[0.94, 0.3, 0], lw=2,
              label='backward stepwise')
plt.legend(handles=[h3[0], h1[0], h2[0]], loc=4)
plt.xlabel('Number of securities')
ticks = np.arange(0, 10 * (n_ // 10 + 1), 10)
plt.xticks(np.append(1, np.append(ticks, n_)))
plt.xlim([0.5, n_])
plt.ylabel('-Te')
plt.title('n-choose-k routines comparison', fontweight='bold')
add_logo(fig, location=5)
plt.tight_layout()
