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

# # S_FitPoissonGMMFP [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_FitPoissonGMMFP&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerGMMpoiss).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import arange, where, round, mean, r_, unique, array
from numpy import min as npmin, max as npmax

from scipy.stats import norm, poisson
from scipy.io import loadmat

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, bar, legend, xlim, ylim, scatter, ylabel, \
    title, xticks, yticks

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, date_mtop, struct_to_dict
from HistogramFP import HistogramFP
from ColorCodedFP import ColorCodedFP
from IterGenMetMomFP import IterGenMetMomFP
from binningHFseries import binningHFseries
from BlowSpinFP import BlowSpinFP
# -

# ## Upload database

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_US_10yr_Future_quotes_and_trades'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_US_10yr_Future_quotes_and_trades'), squeeze_me=True)

trades = struct_to_dict(db['trades'])
# -

# ## Comupte the invariants: eps= dn = number of trades in 1-second time intervals

# +
flag_dt = '1second'
trade = unique(trades.time)  # trade time
dk, k,*_ = binningHFseries(trade, flag_dt)
time = array([date_mtop(i) for i in arange(trade[0], trade[-1], 1.1574e-05)])
# time = cellstr((time))
time = time[1:]

epsi = dk
t_ = epsi.shape[1]
# -

# ## Set a Flexible Probabilities scenario using BlowSpinFP

U = norm.rvs(mean(epsi), 1, size=(1, t_))
b = 1
s = 0
FP1, ens1 = BlowSpinFP(r_[epsi, U], b, s, spinscale=0.2, method='direct')
flex_probs = FP1[[b-1 + s],:]
ens = ens1[b-1 + s]

# ## Estimate the Poisson distribution parameter lam using GMMFP

Parameter = IterGenMetMomFP(epsi, flex_probs, 'Poisson', 2)
lam = Parameter.lam

# ## HFP histogram

# +
options = namedtuple('option', 'n_bins')
options.n_bins = t_
p, x = HistogramFP(epsi, flex_probs, options)
p = p*(x[1] - x[0])  # normalization (such that sum(p)==1)

j = where(p[0] == 0)[0]
p[0,j] = np.NaN
x[j] = np.NaN
x = round(x)
# -

# ## Fitted Poisson pdf

fitted_pdf = poisson.pmf(x[~np.isnan(x)], lam)
x = x[~np.isnan(x)]
p = p[0,~np.isnan(p[0])].reshape(1,-1)

# ## Generate the figure

myFmt = mdates.DateFormatter('%H:%M:%S')
f = figure()
# HFP histogram with fitted pdf superimposed
ax = plt.subplot2grid((10,1),(0,0),rowspan=4)
ax.set_facecolor('white')
b = bar(x[:-1], p[0], width=x[1]-x[0],facecolor=[.8, .8, .8], edgecolor=[.6, .6, .6])
bb = plot(x, fitted_pdf, marker='.')
xlim([npmin(x), npmax(x)])
ylim([0, max(npmax(p), npmax(fitted_pdf))])
yticks([])
P1 = 'Fitted Poisson pdf( $\lambda$=%3.2f)' % lam
legend([P1,'HFP distr.'])
# Scatter plot of the tick-time increments with color-coded observations (according to the FP)
CM, C = ColorCodedFP(flex_probs, npmin(flex_probs), npmax(flex_probs), arange(0,0.71,0.01), 0, 1, [1, 0])
ax = plt.subplot2grid((10,1),(4,0),rowspan=3)
# colormap(CM)
scatter(time, epsi[0], s=15, c=C, marker='.',cmap=CM)
xlim([min(time), max(time)])
xticks(time[arange(59,t_-1,120)])
ax.xaxis.set_major_formatter(myFmt)
ax.set_facecolor('white')
ylim([npmin(epsi), npmax(epsi)])
ylabel('Tick-time increments')
# Flexible Probabilities profile
ax = plt.subplot2grid((10,1),(7,0),rowspan=3)
bb = bar(time,flex_probs[0],width=time[1].toordinal()-time[0].toordinal(),facecolor=[.7, .7, .7], edgecolor='k')
xlim([min(time),max(time)])
plt.xticks(time[arange(59,t_-1,120)])
plt.yticks([])
ax.xaxis.set_major_formatter(myFmt)
ax.set_facecolor('white')
ylim([0, 1.3*npmax(flex_probs)])
ensT = 'Effective Num.Scenarios =  %3.0f' %ens
plt.text(time[60], 1.1*npmax(flex_probs), ensT, color='k',horizontalalignment='left',verticalalignment='bottom')
title('Flexible Probabilities')
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
