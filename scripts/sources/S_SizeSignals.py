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

# # S_SizeSignals [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_SizeSignals&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-size-signal).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import arange, array, argsort, linspace, diag, round, log, exp, sqrt, zeros, sum as npsum

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, xlim, ylim, subplots, title

plt.style.use('seaborn')
np.seterr(invalid='ignore')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, date_mtop
from FPmeancov import FPmeancov
from EwmaFP import EwmaFP
# -

# ## Upload the database

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_strategies'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_strategies'), squeeze_me=True)

last_price = db['last_price']
dates = db['dates']
s_siz = db['s_siz']

# settings
v = last_price
[n_, t_] = v.shape
t_start = 252*2  # starting point of the strategy
# -

# ## Compute the smoothed signals

# +
tauHL_smoo = log(2) / 10
t_smoo = 180
s_siz_smoo = zeros((n_,t_-t_smoo+1))

for t in range(t_smoo,s_siz.shape[1]+1):
    s_siz_smoo[:, [t - t_smoo]] = EwmaFP(s_siz[:, t - t_smoo :t], tauHL_smoo)[0]
# -

# ## Use the smoothed signals to compute the scored signal

t_scor = 252
s_siz_scor = zeros((n_,s_siz_smoo.shape[1]-t_scor+1))
tauHL_scor = log(2) / 120
p_scor = exp(-tauHL_scor*arange(t_scor - 1, 0 + -1, -1)).reshape(1,-1) / npsum(exp(-tauHL_scor*arange(t_scor - 1, 0 + -1, -1)))
for t in arange(t_scor,s_siz_smoo.shape[1]+1):
    mu_siz, cov_siz = FPmeancov(s_siz_smoo[:, t - (t_scor):t], p_scor)
    s_siz_scor[:, t - t_scor] = (s_siz_smoo[:,t-1] - mu_siz.flatten()) / sqrt(diag(cov_siz))

# ## Use the scored signals to compute the ranked signals

s_siz_rk = zeros((n_,s_siz_scor.shape[1]))
for t in range(s_siz_scor.shape[1]):
    rk = argsort(s_siz_scor[:,t])
    rk_signal = argsort(rk)+1
    s_siz_rk[:,t] = (rk_signal - 0.5*n_)*(2 / n_)

# ## Compare the plots of one signal, one smoothed signal and one scored signal

dates = dates[t_start-1:]
grid_dates = linspace(0, len(dates)-1, 5)
grid_dates = list(map(int,round(grid_dates)))  # integer numbers
index = argsort(s_siz_rk[:,-1])

# ## Compare the plots of a cluster of 4 scored signals with their ranked counterparts

dates_dt = array([date_mtop(i) for i in dates])
date_tick = dates_dt[grid_dates]
f, ax = subplots(2,1)
plt.sca(ax[0])
xx = t_start-1
plot(dates_dt, s_siz[index[int(round(n_*0.2))-1], xx:])
plt.xticks(dates_dt[grid_dates])
xlim([dates_dt[0], dates_dt[-1]])
title('Size versus smoothed size signal')
xx = t_start - t_smoo
plot(dates_dt, s_siz_smoo[index[int(round(n_*0.2))-1], xx:], 'r')
plt.xticks(dates_dt[grid_dates])
xlim([dates_dt[0], dates_dt[-1]])
plt.sca(ax[1])
xx = t_start - t_smoo - t_scor+1
plot(dates_dt, s_siz_scor[index[int(round(n_*0.2))-1], xx:])
plt.xticks(dates_dt[grid_dates])
xlim([dates_dt[0], dates_dt[-1]])
title('Scored size signal')
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
f, ax = subplots(2,1)
plt.sca(ax[0])
plot(dates_dt, s_siz_scor[[index[int(round(n_*0.2))-1], index[int(round(n_*0.4))-1], index[int(round(n_*0.6))-1],index[int(round(n_*0.8))-1]],xx:].T)
plt.xticks(dates_dt[grid_dates])
xlim([dates_dt[0], dates_dt[-1]])
title('Scored size signal cluster')
plt.sca(ax[1])
plot(dates_dt,s_siz_rk[[index[int(round(n_*0.2))-1], index[int(round(n_*0.4))-1], index[int(round(n_*0.6))-1], index[int(round(n_*0.8))-1]], xx:].T)
plt.xticks(dates_dt[grid_dates])
xlim([dates_dt[0], dates_dt[-1]])
ylim([-1.05, 1.05])
title('Ranked size signal cluster')
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
