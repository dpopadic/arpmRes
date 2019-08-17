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

# # S_BackForwEwmaSD [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_BackForwEwmaSD&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-estimation-fwd-bwd-exp-smooth).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import arange, array, zeros, diff, abs, log, exp, sqrt, linspace
from numpy import sum as npsum

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots, title
import matplotlib.dates as mdates

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict, date_mtop

# Parameters
tau_HL = 30
lam = log(2) / tau_HL
i_ = 252
# -

# ## Upload database db_Stock SPX

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_Stocks'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_Stocks'), squeeze_me=True)

SPX = struct_to_dict(db['SPX'])
# -

# ## Compute the realized compounded returns

# +
v = SPX.Price_close

x = log(v)
epsi = diff(x, 1).reshape(1,-1)
date = SPX.Date[1:]

t_ = epsi.shape[1]
# -

# ## Compute the backward-forward exponential decay probabilities

edecayprobs = exp(-lam*(abs(arange(-i_, i_ + 1)))).reshape(1,-1)
gamma = npsum(edecayprobs)  # normalization coefficient
edecayprobs = edecayprobs / gamma  # decay factors

# ## Compute the backward/forward exponentially weighted moving standard deviations

y = zeros(t_ - 2 * i_)  # start from time= i_+1 and estimate up to time= t_end -i_  (so that i_ observations are always availabe both backward and forward)
for t in arange(i_,t_-i_):
    ret = epsi[[0],t - i_:t + i_+1]
    y[t - i_] = sqrt(edecayprobs@ret.T ** 2)

# ## Display the compounded returns and the backward/forward exponentially weighted moving standard deviations

# +
date_dt = array([date_mtop(i) for i in date])
myFmt = mdates.DateFormatter('%d-%b-%Y')

f, ax = subplots(2, 1)
date_est = date_dt[i_:t_- i_]
ax[0].plot(date_est, epsi[0,i_:t_ - i_], color='b',lw=1)
ax[0].set_xlim([date_est[0], date_est[-1]])
ax[0].xaxis.set_major_formatter(myFmt)
title('Compounded returns')

date_est = date_dt[i_ :t_- i_]
ax[1].plot(date_est, y, color=[.9, .4, 0], lw = 1.5)
ax[1].set_xlim([date_est[0], date_est[-1]])
ax[1].xaxis.set_major_formatter(myFmt)
title('Estimated Exponentially Weighted Moving St. Deviation')
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
# -

# ## Display the backward/forward exponential decay probabilities

f, ax = subplots(1, 1)
ax.bar(arange(edecayprobs.shape[1]),edecayprobs[0], facecolor=[.7, .7, .7], edgecolor=[.7, .7, .7])
ax.set_xlim([1, 2 * i_ + 1])
plt.xticks(linspace(1,2*i_+1,3),[-252,0,252])
title('Exponential decay factors profile');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

