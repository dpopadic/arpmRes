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

# # S_FlexProbDirichlet [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_FlexProbDirichlet&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=exer-fpspec-copy-1).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

from numpy import arange, array, diff, log, exp
from numpy import sum as npsum, min as npmin, max as npmax

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, bar, xlim, ylim, scatter, ylabel, \
    xlabel, title, xticks, yticks
import matplotlib.dates as mdates

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict, date_mtop
from HistogramFP import HistogramFP
from EffectiveScenarios import EffectiveScenarios
from Stats import Stats
from ColorCodedFP import ColorCodedFP
from Dirichlet import Dirichlet
# -

# ## Upload database

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_Stocks'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_Stocks'), squeeze_me=True)

SPX = struct_to_dict(db['SPX'])
# -

# ## Compute the realized time series of the S&P 500 log-returns

# +
SPX_ = SPX.Price_close
date = SPX.Date
epsi = diff(log(SPX_))

t_ = len(epsi)
epsi = epsi.reshape(1,-1)
date = date[1:]
# -

# ## FLEXIBLE PROBABILITIES FROM DIRICHLET DISTRIBUTION

# +
typ = namedtuple('type','Entropy')
typ.Entropy = 'Exp'

lam = 0.0015
p0 = exp(-lam*arange(t_, 1 + -1, -1)).reshape(1,-1)
p0 = p0 / npsum(p0)

prob_dir = Dirichlet(p0*10 ** 4)
prob_dir = prob_dir / npsum(prob_dir)
ens = EffectiveScenarios(prob_dir, typ)
# -

# ## HFP histogram and statistics

# +
option = namedtuple('option', 'n_bins')
option.n_bins = 10*log(epsi.shape[1])
p, x = HistogramFP(epsi, prob_dir, option)

mu, sdev, VaR, CVaR, skewness, kurtosis = Stats(epsi, prob_dir)
# -

# ## Figure

# +
date_tick = arange(99, t_, 680)
date_dt = array([date_mtop(i) for i in date])
xtick = date_dt[date_tick]
myFmt = mdates.DateFormatter('%d-%b-%Y')

figure()

# FP profile
plt.subplot2grid((2,3),(0,0),colspan=2)
bar(date_dt, prob_dir[0], width=date_dt[1].toordinal()-date_dt[0].toordinal(),facecolor=[0.5, 0.5, 0.5], edgecolor=[0.5, 0.5, 0.5])
# colormap((gray))
xlim([min(date_dt), max(date_dt)])
xticks(xtick)
ylim([0, 1.1*npmax(prob_dir)])
yticks([])
plt.gca().xaxis.set_major_formatter(myFmt)
plt.gca().set_facecolor('white')
title('FLEXIBLE PROBABILITIES FROM DIRICHLET DISTRIBUTION')
ylabel('probability')
TEXT = 'Effective Num.Scenarios =  % 3.0f'%ens
plt.text(date_dt[50], 1.05*npmax(prob_dir), TEXT,horizontalalignment='left')

# scatter colormap and colors
[CM, C] = ColorCodedFP(prob_dir, 10 ** -20, max(prob_dir[0]), arange(0,0.85,0.05), 0, 20, [20, 0])

# Time series of S&P500 log-rets
ax = plt.subplot2grid((2,3),(1,0),colspan=2)
# colormap(CM)
scatter(date_dt, epsi, 10, c=C, marker='.', cmap=CM)
xlim([min(date_dt), max(date_dt)])
xticks(xtick)
ax.set_facecolor('white')
ax.xaxis.set_major_formatter(myFmt)
ylim([1.1*npmin(epsi),1.1*npmax(epsi)])
ylabel('returns')
title('S&P')

# HFP histogram
ax = plt.subplot2grid((2,3),(1,2))
plt.barh(x[:-1], p[0], height=x[1]-x[0], facecolor=[0.7, 0.7, 0.7], edgecolor=[0.5, 0.5, 0.5])
xlim([0, 1.05*npmax(p)])
ax.set_facecolor('white')
xticks([])
yticks([])
ylim([1.1*npmin(epsi), 1.1*npmax(epsi)])
xlabel('probability')

# statistics
TEXT = 'Mean  % 3.3f \nSdev    %3.3f \nVaR      %3.3f \nCVaR   %3.3f \nSkew   %3.3f \nKurt     %3.3f '%(mu,sdev,VaR,CVaR,skewness,kurtosis)
plt.text(0.5*npmax(p), 0.025, TEXT,horizontalalignment='left',verticalalignment='bottom')
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
