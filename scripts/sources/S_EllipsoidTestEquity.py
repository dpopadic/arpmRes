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

# # S_EllipsoidTestEquity [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_EllipsoidTestEquity&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=IIDtestEquity).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict
from autocorrelation import autocorrelation
from Price2AdjustedPrice import Price2AdjustedPrice
from InvarianceTestEllipsoid import InvarianceTestEllipsoid
# -

# ## Upload the database

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_Stocks'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_Stocks'), squeeze_me=True)

StocksSPX = struct_to_dict(db['StocksSPX'])
# -

# ## Compute the dividend adjusted prices of one stock

# +
stock_index = 1
v = StocksSPX.Prices[[stock_index-1], :]
date = StocksSPX.Date
div = StocksSPX.Dividends[stock_index-1]
if div.size != 0:
    v, _ = Price2AdjustedPrice(date.reshape(1,-1), v, div)

lag_ = 10
# -

# ## Compute the time series of each variable

x = v[[0],1:] / v[[0],:-1]
y = v[[0],1:] - v[[0],:-1]
z = (v[[0],1:] / v[[0],:-1]) ** 2
w = v[[0],2:] - 2 * v[[0],1:-1] + v[[0],:-2]

# ## Compute the autocorrelations of each variable

acf_x = autocorrelation(x, lag_)
acf_y = autocorrelation(y, lag_)
acf_z = autocorrelation(z, lag_)
acf_w = autocorrelation(w, lag_)

# ## Plot ellipsoid and auto correlation coefficients

# +
ell_scale = 2  # ellipsoid radius coefficient
fit = 0  # fitting

lag = 10  # lag to be printed in the plots

# x
for plotvar, acfvar, varname in zip([x, y, z, w], [acf_x, acf_y, acf_z, acf_w], ['X', 'Y', 'Z', 'W']):
    f = figure(figsize=(12,6))
    InvarianceTestEllipsoid(plotvar, acfvar[0,1:], lag, fit, ell_scale, None, 'Invariance Test ({var})'.format(var=varname));
    # save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
