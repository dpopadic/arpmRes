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

# # S_UnconditionalEstimation [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_UnconditionalEstimation&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ex-rel-entropy-estim-mfp).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import arange, array, linspace, round, log, exp, sqrt
from numpy import sum as npsum, max as npmax

from scipy.stats import t
from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, bar, legend, xlim, ylim, ylabel, \
    title
import matplotlib.dates as mdates

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, date_mtop
from HistogramFP import HistogramFP
from EffectiveScenarios import EffectiveScenarios
from FitSkewtMLFP import FitSkewtMLFP
# -

# ## Upload database

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_MomStratPL'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_MomStratPL'), squeeze_me=True)

dailypnl = db['dailypnl']
dates = db['dates']
# -

# ## Select data and set flexible probabilities

# +
y = dailypnl  # select observations
t_ = len(dates)

lam = log(2) / 180
p = exp(-lam *arange(len(y),0,-1)).reshape(1,-1)
p = p /npsum(p)  # FP-profile: exponential decay 6 months
typ = namedtuple('type','Entropy')
typ.Entropy = 'Exp'
ens = EffectiveScenarios(p, typ)  # effective number of scenarios
# -

# ## Compute MFP-estimators of location and scatter by using Relative Entropy formulation of ML principle

mu, sigma, alpha, nu = FitSkewtMLFP(y[np.newaxis,...], p)

# ## Compute the FP-histogram approximating the unconditional pdf and evaluate the estimated Skew t distribution

# +
option = namedtuple('option', 'n_bins')

option.n_bins = round(7*log(ens))
hgram, x_hgram = HistogramFP(y[np.newaxis,...], p, option)
x_m = min(x_hgram)
x_M = max(x_hgram)
x_mM = x_M - x_m

x = linspace(x_m-.15*x_mM, x_M+.15*x_mM, 101)
x_bar = (x - mu) / sigma
x_tilde = alpha * x_bar * sqrt(nu + 1) / (nu + x_bar ** 2)
skewt = (2 / sigma) * t.pdf(x_bar, nu) * t.cdf(x_tilde, (nu + 1))  # Skew t pdf
# -

# ## Generate figure showing the HFP-pdf with the Skew t-fit superimposed

# +
y_m = min(y)
y_M = max(y)
y_mM = y_M - y_m
d = linspace(0,t_-1,4, dtype=int)
dates_dt = array([date_mtop(i) for i in dates])

f = figure()
myFmt = mdates.DateFormatter('%d-%b-%Y')
ax = plt.subplot2grid((3,3),(2,0),colspan=2)
plt.sca(ax)
# Flexible Probability profile
wid = dates_dt[1].toordinal()-dates_dt[0].toordinal()
b = bar(dates_dt, p[0], width=wid, facecolor=[.7, .7, .7], edgecolor=[.7, .7, .7])
xlim([min(dates_dt), max(dates_dt)])
ylim([0, npmax(p)])
plt.xticks(dates_dt[d])
plt.yticks([])
ylabel('FP')
ax.xaxis.set_major_formatter(myFmt)
ensT = 'Effective Num.Scenarios =  %3.0f'%ens
plt.text(0.05, 0.8, ensT, horizontalalignment='left',verticalalignment='bottom',transform=ax.transAxes)
title('Flexible Probabilities - Exponential Decay');

ax = plt.subplot2grid((3,3),(2,2))
plt.sca(ax)
plt.axis('off')
stat1 = ' $\mu$ =  %1.3e\n $\sigma$ = %1.3e \n $\\alpha$ = %1.3e \n $\\nu$ =  %1.3e '%(mu,sigma,alpha,nu)
plt.text(0.5, 0.5, stat1, horizontalalignment='center',verticalalignment='center',multialignment='left',transform=ax.transAxes)

# pdf plot
ax = plt.subplot2grid((3,3),(0,0),colspan=3,rowspan=2)
plt.sca(ax)
b_1 = bar(x_hgram[:-1], hgram[0], width=x_hgram[1]-x_hgram[0], facecolor=[.7, .7, .7], edgecolor=[.3, .3, .3])
plt.axis([x_m - .15*x_mM, x_M + .15*x_mM, 0, npmax(hgram) + (npmax(hgram) / 20)])
l = plot(x, skewt, color=[.9, .3, 0], lw=1.5)
legend(['Skew t fit','Unconditional pdf'])
title('Relative Entropy Estimation for MFP Unconditional Properties')
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
