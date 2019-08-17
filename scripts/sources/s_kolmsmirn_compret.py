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

# # s_kolmsmirn_compret [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_kolmsmirn_compret&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=exer-comp-rets-copy-1).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arpym.statistics import invariance_test_ks
from arpym.tools import adjusted_value, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_kolmsmirn_compret-parameters)

# +
t_first = '25-2-2010'  # starting date
t_last = '17-7-2012'  # ending date
fwd = True  # indicator for forward of backward adjusted value
conf_lev = 0.95  # confidence level
# -

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_kolmsmirn_compret-implementation-step00): Load data

# +
# loading data from 03-01-1994 to 24-11-2017
path = '../../../databases/global-databases/equities/db_stocks_SP500/'
df_nokia_stock = pd.read_csv(path + 'NOK_prices.csv',
                             header=0)
df_nok_dividends = pd.read_csv(path + 'NOK_dividends.csv',
                               header=0)

# convert column 'date' from string to datetime64
df_nokia_stock['date_tmstmp'] = pd.to_datetime(df_nokia_stock.date,
                                               dayfirst=True)
df_nok_dividends['date_tmstmp'] = pd.to_datetime(df_nok_dividends.date,
                                                 dayfirst=True)

t_first = pd.to_datetime(t_first, dayfirst=True)
t_last = pd.to_datetime(t_last, dayfirst=True)
# filter the data for the selected range
nok_stock_long = df_nokia_stock[(df_nokia_stock.date_tmstmp >= t_first) &
                                (df_nokia_stock.date_tmstmp < t_last)]
nok_dividends = df_nok_dividends[(df_nok_dividends.date_tmstmp >= t_first) &
                                 (df_nok_dividends.date_tmstmp < t_last)]
# extract values
dates = nok_stock_long.date_tmstmp.values
v_stock = nok_stock_long.close.values
r = nok_dividends.date_tmstmp.values
cf_r = nok_dividends.dividends.values
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_kolmsmirn_compret-implementation-step01): Dividend-adjusted values and invariant series

# +
v_adj = adjusted_value(v_stock, dates, cf_r, r, fwd)
epsi = np.diff(np.log(v_adj))
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_kolmsmirn_compret-implementation-step02): Perform Kolmogorov-Smirnov test

# +
plt.style.use('arpm')

# perform and show Kolmogorov-Smirnov test for invariance
z_ks, z = invariance_test_ks(epsi, conf_lev=conf_lev)
fig = plt.gcf()
add_logo(fig, set_fig_size=False, size_frac_x=1/8)
# -
