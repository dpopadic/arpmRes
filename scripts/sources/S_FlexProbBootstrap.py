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

# # S_FlexProbBootstrap [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_FlexProbBootstrap&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerFPspec).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

from numpy import arange, array, zeros, diff, log
from numpy import min as npmin, max as npmax
from numpy.random import choice

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

# ## FLEXIBLE PROBABILITIES FROM BOOTSTRAP

# +
k_ = 252  # size of subsamples
q_ = 5  # number of subsamples (and frames)

prob_bs = zeros((q_, t_))

ens = zeros((1, q_))
typ = namedtuple('type','Entropy')
typ.Entropy = 'Exp'

for q in range(q_):
    r = choice(arange(t_), size=k_, replace=False)
    prob_bs[q, r] = 1 / k_
    ens[0,q] = EffectiveScenarios(prob_bs[[q],:], typ)
# -

# ## HFP histogram and statistics

# +
q_ = prob_bs.shape[0]
option = namedtuple('option', 'n_bins')
option.n_bins = 10*log(epsi.shape[1])
p, x = {}, {}
for q in range(q_):
    p[q], x[q] = HistogramFP(epsi, prob_bs[[q],:], option)

mu, sdev, VaR, CVaR, skewness, kurtosis = Stats(epsi, prob_bs)
# -

# ## Figure

date_tick = arange(99, t_-1, 680)
date_dt = array([date_mtop(i) for i in date])
myFmt = mdates.DateFormatter('%d-%b-%Y')

# ## q=0

for q in range(2):
    figure()

    # FP profile
    plt.subplot2grid((3, 3), (0, 0), colspan=2)
    plt.gca().set_facecolor('white')
    bar(date_dt, prob_bs[q, :], facecolor=[0.5, 0.5, 0.5], edgecolor=[0.5, 0.5, 0.5])
    xlim([min(date_dt), max(date_dt)])
    xticks(date_dt[date_tick])
    plt.gca().xaxis.set_major_formatter(myFmt)
    ylim([0, 1.1 * npmax(prob_bs[q, :])])
    yticks([])
    title('FLEXIBLE PROBABILITIES FROM BOOTSTRAP')
    ylabel('probability')
    TEXT = 'Effective Num.Scenarios = % 3.0f' % ens[0, q]
    plt.text(min(date_dt), 1.05 * npmax(prob_bs[q, :]), TEXT, horizontalalignment='left')

    # scatter colormap and colors
    CM, C = ColorCodedFP(prob_bs[[q], :], 10 ** -20, npmax(prob_bs[:5, :]), arange(0, 0.95, 0.05), 0, 1, [1, 0])

    # Time series of S&P500 log-rets
    ax = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
    scatter(date_dt, epsi, 15, c=C, marker='.', cmap=CM)
    xlim([min(date_dt), max(date_dt)])
    xticks(date_dt[date_tick])
    plt.gca().xaxis.set_major_formatter(myFmt)
    ax.set_facecolor('white')
    ylim([1.1 * npmin(epsi), 1.1 * npmax(epsi)])
    ylabel('returns')
    title('S&P')

    # HFP histogram
    plt.subplot2grid((3, 3), (1, 2), rowspan=2)
    plt.gca().set_facecolor('white')
    plt.barh(x[q][:-1], p[q][0], height=x[q][1] - x[q][0], facecolor=[0.7, 0.7, 0.7], edgecolor=[0.5, 0.5, 0.5])
    xlim([0, 1.05 * npmax(p[q])])
    xticks([])
    yticks([]), ylim([1.1 * npmin(epsi), 1.1 * npmax(epsi)])
    xlabel('probability')
    plt.tight_layout();
    # statistics
    TEXT = 'Mean  % 3.3f \nSdev    %3.3f \nVaR      %3.3f \nCVaR   %3.3f \nSkew   %3.3f \nKurt     %3.3f' % (
        mu[q], sdev[q], VaR[q], CVaR[q], skewness[q], kurtosis[q])
    plt.text(0.5 * npmax(p[q]), 0.08, TEXT, horizontalalignment='left', verticalalignment='bottom');
    # save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
