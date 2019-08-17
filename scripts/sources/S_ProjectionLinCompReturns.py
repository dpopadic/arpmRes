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

# # S_ProjectionLinCompReturns [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_ProjectionLinCompReturns&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-eq-linvs-comp-proj-ret).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import arange, array, zeros, std, diff, linspace, mean, exp, sqrt, r_
from numpy import min as npmin, max as npmax

from scipy.stats import norm, lognorm
from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, xlim, ylim, subplots, ylabel, \
    xlabel, title

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict
from Price2AdjustedPrice import Price2AdjustedPrice
# -

# ## Upload stock prices from db_Stocks

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_Stocks'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_Stocks'), squeeze_me=True)

StocksSPX = struct_to_dict(db['StocksSPX'])
# -

# ## Compute compounded returns  from dividend adjusted prices

[_, c] = Price2AdjustedPrice(StocksSPX.Date.reshape(1,-1), StocksSPX.Prices[[1]], StocksSPX.Dividends[1])  # Exxon Mobil Corporation

# ## Estimate the parameters((mu,sigma))of the invariants  under the normality assumption.

mu = mean(c)
sigma = std(c,ddof=1)

# ## Compute the distribution of compounded and linear returns at horizons tau

# +
# Set projection parameters
tau = arange(63,600,63)
p_lev = array([.01, .99])
l_ = 100
scale = 0.7*npmin(diff(tau))

x_c = {}
y_c = {}
x_l = {}
y_l = {}

q_c = zeros((len(p_lev), len(tau)))
q_l = zeros((len(p_lev), len(tau)))

for k in range(len(tau)):
    # compounded returns
    q_c[:,k] = norm.ppf(p_lev, mu*tau[k], sigma*sqrt(tau[k]))
    x_c[k] = linspace(npmin(q_c[:,k])-0.4, npmax(q_c[:,k])+0.4,l_)
    y_c[k] = norm.pdf(x_c[k], mu*tau[k], sigma*sqrt(tau[k]))
    y_c[k] = scale*y_c[k] / max(y_c[k])

    # linear returns
    q_l[:,k] = exp(q_c[:,k])-1
    x_l[k] = linspace(npmin(q_l[:,k])-0.4, npmax(q_l[:,k])+0.4,l_)
    y_l[k] = lognorm.pdf(x_l[k] + 1, sigma*sqrt(tau[k]), scale=exp(mu*tau[k]))
    y_l[k] = scale*y_l[k] / max(y_l[k])
# -

# ## Create  a figure showing the pdf of both linear and compounded returns at certain points in the future
# ## and print the quantiles at the confidence levels 0.01 and 0.99.

# +
col = [.8, .8, .8]

f, ax = subplots(2,1)
plt.sca(ax[0])
plot(r_[0, tau], r_['-1',zeros((2,1)), q_c].T, color='r')
for k in range(len(tau)):
    xx =r_[tau[k], tau[k]+y_c[k].T, tau[k]]
    yy =r_[x_c[k][0], x_c[k].T, x_c[k][-1]]
    plt.fill_between(xx, yy, color=col)
xlim([0, npmax(xx)*1.01])
ylim([npmin(yy)*1.2, npmax(yy)*1.2])
xlabel('horizon (years)')
ylabel('return range')
plt.xticks(r_[0,tau],r_[0,tau]/252)
plt.grid(True)
title('Compounded return propagation')
plt.sca(ax[1])
plot(r_[0, tau], r_['-1',zeros((2,1)), q_l].T, color='r')
for k in range(len(tau)):
    xx =r_[tau[k], tau[k]+y_l[k].T, tau[k]]
    yy =r_[x_l[k][0], x_l[k].T, x_l[k][-1]]
    plt.fill_between(xx, yy, color=col)
xlim([0, npmax(xx)*1.01])
ylim([npmin(yy)*1.1, npmax(yy)*1.1])
xlabel('horizon (years)')
ylabel('return range')
plt.xticks(r_[0,tau],r_[0,tau]/252)
plt.grid(True)
title('Linear return propagation')
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

