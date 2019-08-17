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

# # S_EnsembleFlexProbs [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_EnsembleFlexProbs&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerEnsembleFP).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import arange, array, ones, diff, abs, log, exp, sqrt, r_
from numpy import sum as npsum, min as npmin, max as npmax

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, bar, legend, xlim, ylim, scatter, ylabel, \
    xlabel, title, xticks, yticks
import matplotlib.dates as mdates

plt.style.use('seaborn')
np.seterr(all='ignore')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict, date_mtop
from intersect_matlab import intersect
from HistogramFP import HistogramFP
from RollPrices2YieldToMat import RollPrices2YieldToMat
from EffectiveScenarios import EffectiveScenarios
from ConditionalFP import ConditionalFP
from Stats import Stats
from ColorCodedFP import ColorCodedFP
# -

# ## Upload data

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_Stocks'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_Stocks'), squeeze_me=True)

SPX = struct_to_dict(db['SPX'])
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_VIX'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_VIX'), squeeze_me=True)

VIX = struct_to_dict(db['VIX'])

try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_SwapCurve'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_SwapCurve'), squeeze_me=True)

DF_Rolling = struct_to_dict(db['DF_Rolling'])
# -

# ## Recover the invariants and the time series of the conditioning variables

# +
# invariants (S&P500 returns)
epsi = diff(log(SPX.Price_close))

# CONDITIONING VARIABLES
# 1) VIX (VIX.value)

# 2) 5years Swap Zero Rate
ZeroRates,_ = RollPrices2YieldToMat(DF_Rolling.TimeToMat, DF_Rolling.Prices)
zr5 = ZeroRates[DF_Rolling.TimeToMat == 5,:]

# merging datasets
date,_,_ = intersect(intersect(SPX.Date[1:], VIX.Date), DF_Rolling.Dates)
_, i_spx,_ = intersect(SPX.Date[1:], date)
_, i_vix,_ = intersect(VIX.Date, date)
_, i_zr,_ = intersect(DF_Rolling.Dates, date)

epsi = epsi[i_spx].reshape(1,-1)
z1 = VIX.value[i_vix].reshape(1,-1)
z2 = zr5[0,i_zr].reshape(1,-1)
t_ = len(date)
# -

# ## Compute the Flexible Probabilities conditioning on each of the two factors

# +
alpha = 0.3

# prior
lam = log(2) / 1080
prior = exp(-lam*abs(arange(t_, 1 + -1, -1))).reshape(1,-1)
prior = prior / npsum(prior)

# flex. probs conditioned on VIX (z1)

VIXcond = namedtuple('conditioner', ['Series', 'TargetValue', 'Leeway'])
VIXcond.Series = z1
VIXcond.TargetValue = np.atleast_2d(z1[0,-1])
VIXcond.Leeway = alpha
p1 = ConditionalFP(VIXcond, prior)

# flex. probs conditioned on the swap rate (z2)
ZRcond = namedtuple('conditioner', ['Series', 'TargetValue', 'Leeway'])
ZRcond.Series = z2
ZRcond.TargetValue = np.atleast_2d(z2[[0],[-1]])
ZRcond.Leeway = alpha
p2 = ConditionalFP(ZRcond, prior)
# -

# ## Compute the respective Effective Number of Scenarios and the diversity indicator

# +
# effective number of scenarios

typ = namedtuple('type','Entropy')
typ.Entropy = 'Exp'
ens1 = EffectiveScenarios(p1, typ)
ens2 = EffectiveScenarios(p2, typ)

# diversity indicator
rho2_12 = npsum(sqrt(p1*p2))  # overlap: Bhattacharyya coefficient

dd12 = sqrt(1 - rho2_12)  # Hellinger distance

d1 = dd12  # Diversity
d2 = d1
# -

# ## Weights of the Flexible Probabilities Ensemble Posterior

weights = r_[ens1*d1,  ens2*d2]
weights = weights / npsum(weights)

# ## Optimal set of Flex. Probs as log-mixture

opt_p = exp(weights[0]*log(p1) + weights[1]*log(p2))
opt_p = opt_p / npsum(opt_p)
ens_optp = EffectiveScenarios(opt_p, typ)

# ## S&P returns histogram and statistics

option = namedtuple('option', 'n_bins')
option.n_bins = 10*log(epsi.shape[1])
p_eps, x_eps = HistogramFP(epsi, opt_p, option)
m, Sdev, VaR, CVaR, Sk, K = Stats(epsi, opt_p)

# ## Generate the figure

# +
date_tick = arange(99, len(date),380)
date_dt = array([date_mtop(i) for i in date])
myFmt = mdates.DateFormatter('%d-%b-%y')

figure(figsize=(16,10))
# VIX
ax = plt.subplot2grid((2,5),(0,0),colspan=2)
ph0 = ax.plot(date_dt, p1[0],lw=0.5,color='gray')
xticks([])
yticks([])
ax2 = ax.twinx()
ax2.plot(date_dt, z1[0],color= [0, 0, 0.6],lw=0.5)
ph1 =ax2.plot(date_dt, z1[0,-1]*ones(t_),color= 'r', linestyle='--')
xlim([min(date_dt), max(date_dt)])
ax.set_ylim([0, 1.5*npmax(p1)])
ax2.set_ylim([npmin(z1), 1.3*npmax(z1)])
ax2.set_yticks(arange(20,100,20))
ax2.set_ylabel('VIX',color=[0, 0, 0.6])
ax2.grid(False)
LEG = 'target %2.2f'% z1[0,-1]
LEG1 = 'Entr. Pool. Flex. Probs'
legend(handles=[ph1[0],ph0[0]],labels=[LEG, LEG1],loc='upper right')
title('Conditioning variable: VIX')
ENS_text = 'Effective Num.Scenarios =  % 3.0f'% ens1
plt.text(min(date_dt) , npmax(z1)*1.2, ENS_text,horizontalalignment='left')
# 5 YEARS ZERO SWAP RATE
ax = plt.subplot2grid((2,5),(1,0),colspan=2)
ph0=ax.plot(date_dt, p2[0],lw=0.5,color='gray')
yticks([])
xticks([])
ax2 = ax.twinx()
ax2.plot(date_dt, z2[0],color= [0, 0, 0.6],lw=0.5)
ph1=ax2.plot(date_dt, z2[0,-1]*ones(t_),color='r',linestyle='--')
xlim([min(date_dt), max(date_dt)])
ax.set_ylim([0, 1.5*npmax(p2)])
ax2.set_ylim([0.9*npmin(z2), 1.3*npmax(z2)])
ax2.set_ylabel('Swap rate',color=[0, 0, 0.6])
ax2.set_yticks([0.05])
ax2.grid(False)
LEG = 'target %2.3f'% z2[0,-1]
LEG1 = 'Entr. Pool. Flex. Probs'
legend(handles=[ph1[0],ph0[0]],labels=[LEG, LEG1],loc='upper right')
title('Conditioning variable: swap rate.')
ENS_text = 'Effective Num.Scenarios =  % 3.0f'%ens2
plt.text(min(date_dt) , npmax(z2)*1.2, ENS_text, horizontalalignment='left')
# ENSEMBLE FLEXIBLE PROBABILITIES
ax = plt.subplot2grid((2,5),(0,2),colspan=2)
bar(date_dt, opt_p[0], width=(date_dt[1].toordinal()-date_dt[0].toordinal()), facecolor=[0.6, 0.6, 0.6], edgecolor=[0.6, 0.6, 0.6])
xlim([min(date_dt), max(date_dt)]), ylim([0, 1.05*npmax(opt_p)])
yticks([])
xticks(date_dt[date_tick])
ax.xaxis.set_major_formatter(myFmt)
ylabel('probability')
title('ENSEMBLE FLEXIBLE PROBABILITIES')
ENS_text = 'Effective Num.Scenarios = % 3.0f'%ens_optp
plt.text(min(date_dt) , 1.03*npmax(opt_p), ENS_text, horizontalalignment='left')
# S&P returns
ax = plt.subplot2grid((2,5),(1,2),colspan=2)
# scatter colormap and colors
CM, C = ColorCodedFP(opt_p, npmin(opt_p), npmax(opt_p), arange(0,0.85,0.055), 0, 1, [1, 0.1])
ax.set_facecolor('white')
scatter(date_dt, epsi, 20, c=C, marker='.',cmap=CM)
xlim([min(date_dt), max(date_dt)])
ylim([npmin(epsi), npmax(epsi)])
xticks(date_dt[date_tick])
ax.xaxis.set_major_formatter(myFmt)
ylabel('returns')
title('S&P')
# HFP-histogram
ax = plt.subplot2grid((2,5),(1,4))
plt.barh(x_eps[:-1], p_eps[0], height=x_eps[1]-x_eps[0],facecolor=[0.7, 0.7, 0.7], edgecolor=[0.6, 0.6, 0.6])
xlim([0, 1.05*npmax(p_eps)])
ylim([npmin(epsi), npmax(epsi)])
xlabel('probability')
ax.set_xticks([])
# text relative to S&P RETS HIST
TEXT1 = 'Mean  % 3.3f \nSdev    %3.3f \nVaR      %3.3f \nCVaR   %3.3f \nSkew   %3.3f \nKurt     %3.3f' %(m,Sdev,VaR,CVaR,Sk,K)
plt.text(0.45*npmax(p_eps), 0.05, TEXT1, horizontalalignment='left',verticalalignment='bottom')
plt.tight_layout()
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
