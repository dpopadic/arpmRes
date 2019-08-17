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

# # s_projection_stock_hfp [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_projection_stock_hfp&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-proj-stock-hfp).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arpym.estimation import exp_decay_fp
from arpym.tools import histogram_sp, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_projection_stock_hfp-parameters)

stock = 'AMZN'  # S&P 500 company (ticker)
t_now = '2012-01-01'  # current time (date)
t_ = 504  # length of the stock value time series
tau_hl = 180  # half life (days)

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_projection_stock_hfp-implementation-step00): Upload data

# +
path = '../../../databases/global-databases/equities/db_stocks_SP500/'
df_stocks = pd.read_csv(path + 'db_stocks_sp.csv', skiprows=[0], index_col=0)

# set timestamps
df_stocks = df_stocks.set_index(pd.to_datetime(df_stocks.index))

# select data within the date range
df_stocks = df_stocks.loc[df_stocks.index <= t_now].tail(t_)

# select stock
df_stocks = df_stocks[stock]  # stock value
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_projection_stock_hfp-implementation-step01): Compute risk driver

x = np.log(np.array(df_stocks))  # log-value

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_projection_stock_hfp-implementation-step02): HFP distribution of the invariant

epsi = np.diff(x)  # invariant past realizations
p = exp_decay_fp(t_ - 1, tau_hl)  # exponential decay probabilities

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_projection_stock_hfp-implementation-step03): Scenario probability distribution of the log-value at horizon

x_t_hor = x[-1] + epsi  # distribution of the horizon log-value

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_projection_stock_hfp-implementation-step04): Histogram of the log-value at horizon

h, b = histogram_sp(x_t_hor, p=p, k_=10 * np.log(t_ - 1))

# ## Plots

# +
# settings
plt.style.use('arpm')
mydpi = 72.0
colhist = [.75, .75, .75]
coledges = [.3, .3, .3]
fig, ax = plt.subplots()
ax.set_facecolor('white')
plt.bar(b, h, width=b[1]-b[0], facecolor=colhist, edgecolor=coledges)
plt.xlabel('log-value')
plt.xticks()
plt.yticks()

add_logo(fig, location=1)
plt.tight_layout()
