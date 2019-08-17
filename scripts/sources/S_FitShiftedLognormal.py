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

# # S_FitShiftedLognormal [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_FitShiftedLognormal&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerMMSLN_fig).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

from numpy import arange, array, sign, sort, diff, round, log, exp, sqrt, r_, real
from numpy import sum as npsum, min as npmin, max as npmax

from scipy.stats import lognorm
from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, bar, legend, xlim, ylim, scatter, ylabel, \
    title, xticks, yticks
import matplotlib.dates as mdates

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict, date_mtop, datenum
from HistogramFP import HistogramFP
from EffectiveScenarios import EffectiveScenarios
from MMFP import MMFP
from ColorCodedFP import ColorCodedFP
# -

# ## Upload database

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_OptionStrategy'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_OptionStrategy'), squeeze_me=True)

OptionStrategy = struct_to_dict(db['OptionStrategy'])
# -

# ## Compute the realized time series of daily P&L's

# +
cum_pnl = OptionStrategy.cumPL
pnl = diff(cum_pnl)

HFP = namedtuple('HFP', 'Scenarios, FlexProbs')
HFP.Scenarios = pnl

t_ = len(pnl)
t = arange(t_)
date = OptionStrategy.Dates
date = date[1:]
# -

# ## Set the Flexible Probabilities as exponential decay with half life 500 days and compute the Effective Number of Scenarios
lam = log(2) / 500
flex_probs = exp(-lam*arange(t_, 1 + -1, -1)).reshape(1,-1)
flex_probs = flex_probs / npsum(flex_probs)
typ = namedtuple('type','Entropy')
typ.Entropy = 'Exp'
ens = EffectiveScenarios(flex_probs, typ)

# +
# ## Fit the Shifted lognormal model

HFP.FlexProbs = flex_probs
Parameters = MMFP(HFP, 'SLN')
mu = real(Parameters.mu)
sig2 = real(Parameters.sig2)
c = real(Parameters.c)
param = r_[mu,sig2,c]
# -

# ## Recover the HFP histogram

option = namedtuple('option', 'n_bins')
option.n_bins = round(10*log(t_))
p, x = HistogramFP(pnl.reshape(1,-1), flex_probs, option)

# ## Compute the MMFP pdf

# +
xx = sort(x)
xx = r_[xx, npmax(xx) + arange(0.001,0.051,0.001)]
m1 = flex_probs@pnl.T
m3 = flex_probs@((pnl - m1) ** 3).T

sln = lognorm.pdf(sign(m3)*xx - c, sqrt(sig2), scale=exp(mu))  # fitted pdf

date_dt = array([date_mtop(datenum(i)) for i in date])
myFmt = mdates.DateFormatter('%d-%b-%Y')
date_tick = arange(200-1, t_,820)
# -

# ## Generate the figure

# +
f = figure()
# HFP histogram with MMFP pdf superimposed
h1 = plt.subplot(3, 1, 1)
b = bar(x[:-1], p[0], width=x[1]-x[0], facecolor=[.8, .8, .8], edgecolor=[.6, .6, .6])
bb = plot(xx, sln, lw=2)
xlim([npmin(xx), npmax(xx)])
ylim([0, max(npmax(p), npmax(sln))])
yticks([])
P1 = 'Fitted shift.logn.( $\mu$=%3.1f,$\sigma^2$=%3.1f,c=%3.2f)'%(real(mu),real(sig2),real(c))
l=legend([P1, 'HFP distr.'])
# Scatter plot of the pnl with color-coded observations (according to the FP)
[CM, C] = ColorCodedFP(flex_probs, npmin(flex_probs), npmax(flex_probs), arange(0,0.71,0.01), 0, 18, [18, 0])
h3 = plt.subplot(3,1,2)

scatter(date_dt, pnl, 5, c=C, marker='.',cmap=CM)
xlim([min(date_dt), max(date_dt)])
xticks(date_dt[date_tick])
h3.xaxis.set_major_formatter(myFmt)
ylim([min(pnl), max(pnl)])
ylabel('P&L')
# Flexible Probabilities profile
h2 = plt.subplot(3,1,3)
bb = bar(date_dt,flex_probs[0],facecolor=[.7, .7, .7], edgecolor=[.7, .7, .7])
xlim([min(date_dt), max(date_dt)])
yticks([])
xticks(date_dt[date_tick])
h2.xaxis.set_major_formatter(myFmt)
ylim([0, 1.3*npmax(flex_probs)])
ensT = 'Effective Num.Scenarios =  %3.0f'%ens
plt.text(date_dt[60], 1.1*npmax(flex_probs), ensT, color='k',horizontalalignment='left',verticalalignment='bottom')
title('FLEXIBLE PROBABILITIES')
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
