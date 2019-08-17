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

# # s_stock_short_horizon [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_stock_short_horizon&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerStockShort).

# +
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_stock_short_horizon-parameters)

# day, month and the year of the plotted value
day = 2
month = 9
year = 2015

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_stock_short_horizon-implementation-step00): Load data

# loading data from 2015-05-27 to 2015-12-07
path = '../../../databases/global-databases/high-frequency/db_stock_NOK_intraday/'
df_nokia_stock = pd.read_csv(path + 'data.csv',
                             header=0)
# convert column 'date' from string to datetime64
df_nokia_stock['date'] = pd.to_datetime(df_nokia_stock.date, dayfirst=True)

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_stock_short_horizon-implementation-step01): Select the data to be plotted

t_first = dt.datetime(year, month, day, 9, 30)  # starting time
t_last = dt.datetime(year, month, day, 16, 0)  # ending time
# select data
v_stock = df_nokia_stock[(df_nokia_stock.date >= t_first) &
                         (df_nokia_stock.date <= t_last)]

# ## Plots

# +
plt.style.use('arpm')
# extract values from dataframe
t = v_stock.date.values
v_t_stock = v_stock.price.values

number_of_xticks = 6
tick_array = np.linspace(0, t.shape[0]-1, number_of_xticks, dtype=int)
myFmt = mdates.DateFormatter('%H:%M:%S')

fig = plt.figure()
plt.plot_date(t, v_t_stock, '-')
plt.gca().xaxis.set_major_formatter(myFmt)
plt.xticks(t[tick_array])
plt.xlabel('Time')
plt.ylabel('Value')
plt.title(f'NOKIA intraday value on {dt.date(year, month, day)}')
add_logo(fig)
