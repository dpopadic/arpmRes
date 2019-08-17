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

# # S_ProxyStocks [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_ProxyStocks&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-proxy-stocks).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

from numpy import arange, array, ones, zeros, cov, mean, log, r_
from numpy import min as npmin, max as npmax

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, xlim, ylim, ylabel, \
    xticks
import matplotlib.dates as mdates

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict, date_mtop
from ForwardSelection import ForwardSelection
from NonParamCointegrationFP import NonParamCointegrationFP
from ObjectiveR2 import ObjectiveR2
# -

# ## Upload dataset

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_StocksS_P'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_StocksS_P'), squeeze_me=True)

Data = struct_to_dict(db['Data'])
# -

# ## Compute realized time series of the log values

dates = Data.Dates
z = log(Data.Prices)
x_complete = log(Data.SP_index)

# ## Suppose the complete series is made of t_end=1000 obs and only a_=120 obs are available for the S&P500 series

# +
t_ = 1000
dates = dates[- t_ :]
z = z[:, - t_:]
x_complete = x_complete[- t_ :]

a_ = 120
x_available = x_complete[t_ - a_:]
# -

# ## Select the best pool of factors via forward stepwise regression

# +
m_ = z.shape[0]
k_ = 15  # number of factors

data = namedtuple('data', 'covXZ n_')
data.covXZ = cov(r_[x_available.reshape(1,-1),z[:, t_ - a_:]])
data.n_ = 1

# choice of the factors by stepwise regression
[R2, idx, num] = ForwardSelection(arange(m_), data, ObjectiveR2, 1, k_)
factors = idx[k_-1]
# -

# ## Compute the proxy via PCA, adjusting the level

# +
x = r_[x_available.reshape(1,-1), - z[factors, t_ - a_:]]

theta_threshold = 0.01
p = ones((1, a_)) / a_  # flat Flexible Probabilities
c = NonParamCointegrationFP(x, p, 1, theta_threshold)[0]

b = c[1:, 0] / c[0, 0]

proxy = b.T@z[factors, :t_]
level = mean(x_available) - b.T@mean(z[factors, t_ - a_:], 1)

replicating_series = level + proxy
# -

# ## Compute the percentage errors (residuals)

errors = (replicating_series - x_complete) / x_complete

# ## Figure: plot the original series and the replicating one, along with the percentage errors (residuals)

date_tick = arange(39,t_,120)
grey = [.4, .4, .4]
orange = [.9, .35, 0]
dates_dt = array([date_mtop(i) for i in dates])
from matplotlib.ticker import FuncFormatter
myFmt = mdates.DateFormatter('%d-%b-%y')
# proxy
figure()
plot(dates_dt, x_complete, color=grey,lw=1.5)
plot(dates_dt[:t_- a_], replicating_series[:t_ - a_], color='b',lw= 1.5)
plot(dates_dt[t_ - a_:t_], replicating_series[t_ - a_ :t_], color=orange,lw=1.3)
xlim([min(dates_dt), max(dates_dt)])
xticks(dates_dt[date_tick])
ylim([0.99*npmin(x_complete), 1.01*npmax(x_complete)])
ylabel('SP500 log-value')
plt.gca().xaxis.set_major_formatter(myFmt)
plt.text(dates_dt[499], 4.2, 'Out of sample', color='b')
plt.text(dates_dt[899], 4.2, 'Calibration',color=orange)
leg = legend(['Original series','Proxy']);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
# percentage errors/residuals
figure()
plot(dates_dt, zeros(t_), color= [.7, .7, .7])
plot(dates_dt[:t_ - a_], errors[:t_- a_], '.', markersize=4,color='b')
plot(dates_dt[t_ - a_:t_], errors[t_ - a_ :t_], '.',markersize=4,color=[0.9, .35, 0])
xlim([min(dates_dt), max(dates_dt)])
xticks(dates_dt[date_tick])
ylim([-0.015, 0.015])
plt.gca().xaxis.set_major_formatter(myFmt)
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
ylabel('errors / residuals')
plt.text(dates_dt[399], -0.018, 'Out of sample', color='b')
plt.text(dates_dt[889], -0.018, 'Calibration',color=orange);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
