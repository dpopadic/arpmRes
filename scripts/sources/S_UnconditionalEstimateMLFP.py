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

# # S_UnconditionalEstimateMLFP [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_UnconditionalEstimateMLFP&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ex-log-like-estim-mfp).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import arange, sort, where, round, log, exp, sqrt, r_
from numpy import sum as npsum, min as npmin, max as npmax

from scipy.stats import pareto
from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, bar, legend, xlim, title

np.seterr(divide='ignore')
plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot
from HistogramFP import HistogramFP
from EffectiveScenarios import EffectiveScenarios
from MaxLikelihoodFPLocDispT import MaxLikelihoodFPLocDispT
from FitGenParetoMLFP import FitGenParetoMLFP
from HFPquantile import HFPquantile
from QuantileGenParetoMLFP import QuantileGenParetoMLFP
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

# ## Select data and compute autocorrelations

# +
y = dailypnl.reshape(1,-1)  # select observations
t_ = len(dates)
date = dates

lam = log(2) / 180
p = exp(-lam *arange(y.shape[1],0,-1)).reshape(1,-1)
p = p /npsum(p)  # FP-profile: exponential decay 6 months
# -

# ## Estimate unconditional MaxLikelihoodLFP-mean and covariance

# +
nu = 4  # degrees of freedom
tolerance = 10 ** -7  # Set lower tolerance for higher precision

mu_MLFP, sigma2_MLFP,_ = MaxLikelihoodFPLocDispT(y, p, nu, tolerance, 1)
# -

# ## Estimate unconditional MLFP (EVT) quantile

# +
p_bar = 0.1  # probability threshold
p_quant = r_[arange(10**-4,p_bar+10**-4,10**-4), arange(p_bar+0.001,1.001,0.001)].reshape(1,-1) # quantile probability levels
q_HFP = HFPquantile(y, p_quant, p)
y_bar = q_HFP[p_quant == p_bar]  # threshold
# data below the threshold
l_1 = where(y[0] < y_bar)[0]
l_2 = where(p_quant[0] <= p_bar)[0]
y_ex = y_bar - y[[0],l_1]  # dataset of the conditional excess distribution

csi_MLFP, sigma_MLFP = FitGenParetoMLFP(y_ex, p[0,l_1])  # Maximum Likelihood optimization with Generalized Pareto Distribution
f_MLFP = pareto.pdf(sort(y_ex), csi_MLFP, sigma_MLFP, 0)  # estimated pdf

q_MLFP, *_ = QuantileGenParetoMLFP(y_bar, p_bar, csi_MLFP, sigma_MLFP, p_quant[0,l_2])  # MLFP-quantile

q_bt = q_HFP[0,l_2]  # historical quantile below the threshold
# -

# ## Generate figures showing the unconditional MLFP-mean and standard deviation and the estimated unconditional quantile function

# +
typ = namedtuple('type','Entropy')
typ.Entropy = 'Exp'
ens = EffectiveScenarios(p, typ)
option = namedtuple('option', 'n_bins')

option.n_bins = round(30*log(ens))
hgram, x_hgram = HistogramFP(y, p, option)

g = figure()
# unconditional pdf with mean and dispersion superimposed
ax = plt.subplot2grid((6,1), (0,0), rowspan=3)
b = bar(x_hgram[:-1], hgram[0], width=x_hgram[1]-x_hgram[0],facecolor=[.7, .7, .7], edgecolor=[.3, .3, .3])
plt.axis([npmin(x_hgram), npmax(x_hgram), 0, npmax(hgram) + (npmax(hgram) / 20)])
title('P&L unconditional pdf')
stddev_plot = plot(r_[mu_MLFP - sqrt(sigma2_MLFP), mu_MLFP + sqrt(sigma2_MLFP)],[0, 0], color= [.3, .3, .9], lw=7)
mean_plot = plot(r_[mu_MLFP, mu_MLFP], [0, 0.4*10**-7], color= [.9, .3, 0], lw=7)
legend(['Unconditional MLFP-dispersion','Unconditional MLFP-mean'])

# unconditional quantile
y_min = min([npmin(q_bt), npmin(q_MLFP)])
y_max = max([npmax(q_bt), npmax(q_MLFP)])
ax = plt.subplot2grid((6,1), (3,0), rowspan=2)
xlim([0, npmax(p_quant[0,l_2])])
Q_bt = plot(p_quant[0,l_2], q_bt, color= [.3, .3, .9], lw=2)
Q_MLFP = plot(p_quant[0,l_2], q_MLFP, color= [.9, .3, 0], lw=2)
plt.axis([-10 ** -6, p_bar, y_min - .05*(y_max - y_min), y_max + .05*(y_max - y_min)])
title('Unconditional MLFP-quantile approximation')
legend(['Unconditional quantile','MLFP-quantile approximation'])
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
