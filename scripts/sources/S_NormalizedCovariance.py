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

# # S_NormalizedCovariance [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_NormalizedCovariance&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExCorrVSCov).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from datetime import datetime

import numpy as np
from numpy import where, percentile, diff, cov, log, r_

from scipy.io import loadmat

import matplotlib.pyplot as plt

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import struct_to_dict
from intersect_matlab import intersect
from RollPrices2YieldToMat import RollPrices2YieldToMat
# -

# ## Upload data

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_Stocks'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_Stocks'), squeeze_me=True)

SPX = struct_to_dict(db['SPX'])

try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_SwapCurve'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_SwapCurve'), squeeze_me=True)

DF_Rolling = struct_to_dict(db['DF_Rolling'])
# -

# ## Compute the realized time series of S&P500 daily log-returns and the daily changes in the five-year swap rate

# +
# S&P 500 log-returns
prices = SPX.Price_close
DateSP = SPX.Date

# swap rates
mat = DF_Rolling.TimeToMat
rolling_prices = DF_Rolling.Prices
dateSwap = DF_Rolling.Dates

yields,_ = RollPrices2YieldToMat(mat, rolling_prices)
yield5 = yields[mat == 5,:]  # Swap rate with time to mat = 5

# match the db
[dates, i_ret, i_yield] = intersect(DateSP.T, dateSwap.T)
prices = prices[i_ret]
yield5 = yield5[0,i_yield]

# S&P 500 returns
rets = diff(log(prices), 1)
# 5 years swap rate daily changes
y5changes = diff(yield5, 1)
# Dates
dates = dates[1:]
# -

# ## Normalize the series
# ## Compute sample interquartile range of S&P500 returns and changes in 5yr yield during the past period from 1 January 2005 to 31 December 2010

# +
d1 = datetime(2005, 1, 1).toordinal()+366
d2 = datetime(2010, 12, 31).toordinal()+366
idx = where((dates >= d1) & (dates <= d2))

iqr_rets = percentile(rets[idx], 75) - percentile(rets[idx], 25)
iqr_y5ch = percentile(y5changes[idx],75) - percentile(y5changes[idx], 25)

# Normalization
rets_normalized = rets / iqr_rets
y5changes_normalized = y5changes / iqr_y5ch
# -

# ## SAMPLE COVARIANCE

sample_cov = cov(r_[rets[np.newaxis,...], y5changes[np.newaxis,...]])

# ## NORMALIZED COVARIANCE (sample covariance of the normalized series)

normalized_cov = cov(r_[rets_normalized[np.newaxis,...], y5changes_normalized[np.newaxis,...]])

# ## print results

print(sample_cov)
print(normalized_cov)
