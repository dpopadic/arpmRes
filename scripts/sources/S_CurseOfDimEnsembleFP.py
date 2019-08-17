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

# # S_CurseOfDimEnsembleFP [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_CurseOfDimEnsembleFP&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerENSposterior).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import arange, reshape, array, ones, zeros, std, diff, abs, log, exp, sqrt
from numpy import sum as npsum, max as npmax
from numpy.random import rand

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, bar, xlim, ylim, subplots, ylabel, \
    title, xticks, yticks
import matplotlib.dates as mdates

np.seterr(divide='ignore')

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict, date_mtop
from HistogramFP import HistogramFP
from EffectiveScenarios import EffectiveScenarios
from ConditionalFP import ConditionalFP

# -

# ## upload data

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_StocksS_P'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_StocksS_P'), squeeze_me=True)

Data = struct_to_dict(db['Data'])
# -

# ## Compute the returns on the first 200 stocks in the database (conditioning variables)

# +
ret = diff(log(Data.Prices), 1, 1)

ret = ret[:200, :]
date = Data.Dates[1:]
q_ = ret.shape[0]
t_ = ret.shape[1]
# -

# ## Compute the Flexible probabilities conditioned via Entropy Pooling on each factor for different targets' scenarios

# +
print('Computing the conditioned Flexible Probabilities for each factor')
alpha = 0.2

# Prior: exponential decay
lam = 0.001
prior = exp(-lam * abs(arange(t_, 1 + -1, -1))).reshape(1, -1)
prior = prior / npsum(prior)

k_ = 3  # num of different targets

# initialization
flex_probs = ones((q_, t_, k_))
typ = namedtuple('typ', 'Entropy')
typ.Entropy = 'Exp'
target = ones((q_, 1, k_))
ens = zeros((q_, 1, k_))

# flexible probabilities (q=1...Q)

for q in range(q_):
    cond_factor = ret[[q], :]

    # Targets
    for k in range(k_):
        target[q, 0, k] = -2.5 * std(cond_factor) + rand(1) * 5 * std(cond_factor)

    # Conditioned flexible probabilities
    Conditioner = namedtuple('conditioner', ['Series', 'TargetValue', 'Leeway'])
    Conditioner.Series = cond_factor
    Conditioner.TargetValue = target[[q], 0, :]
    Conditioner.Leeway = alpha

    flex_probs[q, :, :] = ConditionalFP(Conditioner, prior).T

    padj = flex_probs[q, :, :]
    for k in range(k_):
        ens[q, 0, k] = EffectiveScenarios(padj[:, [k]].T, typ)  # effective number of scenarios
# -

# ## Ensemble flexible probabilities: compute the final set of FP as a linear mixture or a log-mixture.

# +
rho2 = zeros((q_, q_, k_))
distance = zeros((q_, q_, k_))
diversity = zeros((q_, 1, k_))
weights = zeros((q_, 1, k_))
LinMP = zeros((1, t_, k_))
LogMP = zeros((1, t_, k_))
ensLogMP = zeros(k_)
ensLinMP = zeros(k_)
print('Ensembling the Flexible Probabilities')
for k in range(k_):
    # Battacharayya coeff and Hellinger distances
    for q1 in range(q_):
        for q2 in range(q_):
            rho2[q1, q2, k] = npsum(sqrt(flex_probs[q1, :, k] * flex_probs[q2, :, k]))
            distance[q1, q2, k] = sqrt(abs(1 - rho2[q1, q2, k]))

    # Diversity indicator
    for q in range(q_):
        diversity[q, 0, k] = (1 / (q_ - 1)) * (npsum(distance[q, :, k] - distance[q, q, k]))

    # weights
    weights[:, 0, k] = ens[:, 0, k] * diversity[:, 0, k]
    weights[:, 0, k] = weights[:, 0, k] / npsum(weights[:, 0, k])

    # Linear mixture
    LinMP[0, :, k] = reshape(weights[:, 0, k], (1, q_), 'F') @ flex_probs[:, :, k]
    ensLinMP[k] = EffectiveScenarios(LinMP[[0], :, k], typ)  # effective number of scenarios

    # Log-mixture
    LogMP[0, :, k] = exp(reshape(weights[:, 0, k], (1, q_), 'F') @ log(flex_probs[:, :, k]))
    LogMP[0, :, k] = LogMP[0, :, k] / npsum(LogMP[0, :, k])
    ensLogMP[k] = EffectiveScenarios(LogMP[[0], :, k], typ)  # effective number of scenarios

# computations for the histograms
pflat = ones((1, q_)) / q_
option = namedtuple('option', 'n_bins')
option.n_bins = 10 * log(q_)

nbins = int(option.n_bins)
nW, xW = zeros((nbins, 1, k_)), zeros((nbins + 1, 1, k_))
nE, xE = zeros((nbins, 1, k_)), zeros((nbins + 1, 1, k_))
nD, xD = zeros((nbins, 1, k_)), zeros((nbins + 1, 1, k_))

for k in range(k_):
    nW[:, 0, k], xW[:, 0, k] = HistogramFP(weights[:, 0, [k]].T, pflat, option)
    nE[:, 0, k], xE[:, 0, k] = HistogramFP(ens[:, 0, [k]].T, pflat, option)
    nD[:, 0, k], xD[:, 0, k] = HistogramFP(diversity[:, 0, [k]].T, pflat, option)
# -

# ## Generate figures

date_tick = range(0, len(date), 600)
date_dt = array([date_mtop(i) for i in date])
myFmt = mdates.DateFormatter('%d-%b-%Y')
xtick = date[date_tick]
grey = [0.6, 0.6, 0.6]
blue = [0.2, 0.3, 0.65]
for k in arange(1):
    f1, ax = subplots(2, 1)
    # linear weighted average
    plt.sca(ax[0])
    bar(date_dt, LinMP[0, :, k], facecolor=blue, edgecolor=blue)
    xlim([min(date_dt), max(date_dt)])
    ylim([0, max(LinMP[0, :, k])])
    yticks([])
    xticks(xtick)
    ax[0].xaxis.set_major_formatter(myFmt)
    title('Linear weighted average')
    ylabel('Flexible Prob.')
    T1 = 'Effective Num.Scenarios =  %3.0f' % ensLinMP[k]
    plt.text(date_dt[49], 0.9 * npmax(LinMP[0, :, k]), T1, horizontalalignment='left')
    # non-linear weighted average
    plt.sca(ax[1])
    bar(date_dt, LogMP[0, :, k], facecolor=blue, edgecolor=blue)
    xlim([min(date_dt), max(date_dt)])
    ylim([0, max(LogMP[0, :, k])])
    yticks([])
    xticks(xtick)
    title('Non-linear weighted average')
    ylabel('Flexible Prob.')
    plt.tight_layout();
    T1 = 'Effective Num.Scenarios =  %3.0f' % ensLogMP[k]
    ax[1].xaxis.set_major_formatter(myFmt)
    plt.text(date_dt[49], 0.9 * npmax(LogMP[0, :, k]), T1, horizontalalignment='left')
    # save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
    figure()
    # weights
    ax = plt.subplot2grid((3, 3), (0, 0), colspan=2)
    bar(range(q_), weights[:, 0, k], 1, facecolor=grey, edgecolor=grey)
    xlim([1, q_])
    yticks([])
    xticks(arange(0, q_ + 20, 20))
    ylabel('Weights')
    title('Entries')
    ax = plt.subplot2grid((3, 3), (0, 2))
    plt.barh(xW[:-1, 0, k], nW[:, 0, k], xW[1, 0, k] - xW[0, 0, k], facecolor=grey, edgecolor=grey)
    title('Distribution')
    # Effective Number of Scenarios
    ax = plt.subplot2grid((3, 3), (1, 0), colspan=2)
    bar(range(q_), ens[:, 0, k], 1, facecolor=grey, edgecolor=grey)
    xlim([1, q_])
    yticks([])
    xticks(arange(0, q_ + 20, 20))
    ylabel('Eff. Num. Scen.')
    title('Entries')
    ax = plt.subplot2grid((3, 3), (1, 2))
    plt.barh(xE[:-1, 0, k], nE[:, 0, k], xE[1, 0, k] - xE[0, 0, k], facecolor=grey, edgecolor=grey)
    # ax.set_ylim(yl1)
    title('Distribution')
    # diversity
    ax = plt.subplot2grid((3, 3), (2, 0), colspan=2)
    bar(range(q_), diversity[:, 0, k], 1, facecolor=grey, edgecolor=grey)
    xlim([1, q_])
    yticks([])
    xticks(arange(0, q_ + 20, 20))
    ylabel('Diversity')
    title('Entries')
    ax = plt.subplot2grid((3, 3), (2, 2))
    plt.barh(xD[:-1, 0, k], nD[:, 0, k], xD[1, 0, k] - xD[0, 0, k], facecolor=grey, edgecolor=grey)
    title('Distribution')
    plt.tight_layout();
    # save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
