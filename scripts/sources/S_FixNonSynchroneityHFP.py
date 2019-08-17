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

# # S_FixNonSynchroneityHFP [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_FixNonSynchroneityHFP&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerHFPnonSync).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import arange, array, ones, diff, diag, log, exp, sqrt, r_, zeros
from numpy import sum as npsum

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import xlim, ylim, scatter, subplots, ylabel, \
    xlabel, title, xticks, yticks

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict
from FPmeancov import FPmeancov
from intersect_matlab import intersect
from MinRelEntFP import MinRelEntFP
from EffectiveScenarios import EffectiveScenarios
from Riccati import Riccati
from ColorCodedFP import ColorCodedFP
# -

# ## Upload databases

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_Stocks'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_Stocks'), squeeze_me=True)

SPX = struct_to_dict(db['SPX'])
KOSPI = struct_to_dict(db['KOSPI'])
# -

# ## Compute the log-prices and log-returns of the two indexes

# +
# S&P 500 (US)
NSprice = SPX.Price_close
NSdate = SPX.Date

# KOSPI (Korea)
KSprice = KOSPI.Price_close
KSdate = KOSPI.Date

# merge dataset
[dates, i1, i2] = intersect(NSdate, KSdate)
ret1 = diff(log(NSprice[i1])).reshape(1,-1)
ret2 = diff(log(KSprice[i2])).reshape(1,-1)

t_ = 500
ret1 = ret1[[0],- t_:]
ret2 = ret2[[0],- t_:]
dates = dates[-t_+1:]

epsi = r_[ret1, ret2]
# -

# ## Flexible Probabilities

# +
# flexible prob.
lam = (log(2)) / 120  # half life 4 months
flex_prob = exp(-lam*arange(t_, 1 + -1, -1)).reshape(1,-1)
flex_prob = flex_prob / npsum(flex_prob)

typ = namedtuple('typ', 'Entropy')
typ.Entropy = 'Exp'
ens = EffectiveScenarios(flex_prob, typ)
# -

# ## Twist fix for non-synchroneity in HFP

# +
print('Performing the twist fix for non-synchroneity')
# (step 1-2) HFP MEAN/COVARIANCE/CORRELATION
HFPmu, HFPcov = FPmeancov(epsi, flex_prob)
HFPc2 = np.diagflat(diag(HFPcov) ** (-1 / 2))@HFPcov@np.diagflat(diag(HFPcov) ** (-1 / 2))

# (step 3) TARGET CORRELATIONS
l = 10  # number of lags

flex_prob_l = flex_prob[[0],l:]
flex_prob_l = flex_prob_l / npsum(flex_prob_l)

# concatenate the daily log-returns
y1, y2 = zeros(t_),zeros(t_)
for t in range(l,t_):
    y1[t] = sum(ret1[0,t - l:t])
    y2[t] = sum(ret2[0,t - l:t])

y1 = y1[l:]
y2 = y2[l:]

# compute the correlation
FPstd1 = sqrt(npsum(flex_prob_l * (y1 ** 2)))
FPstd2 = sqrt(npsum(flex_prob_l * (y2 ** 2)))
rho2 = npsum(flex_prob_l * y1 * y2) / (FPstd1*FPstd2)

Target_rho2 = array([[1, rho2], [rho2, 1]])

# (step 4) TARGET COVARIANCES
TargetCOV = np.diagflat(diag(HFPcov) ** (1 / 2))@Target_rho2@np.diagflat(diag(HFPcov) ** (1 / 2))

# (step 5) NEW SCENARIOS [Moment-matching scenario transformation]

# (step 1 [MomMatch routine]) Twist factor
b = Riccati(HFPcov, TargetCOV)

# (step 2-3 [MomMatch routine]) Transform data
new_epsi = b@epsi
# -

# ## Entropy pooling fix for non-synchroneity in HFP

# +
print('Performing the Entropy Pooling fix for non-synchroneity')
# (step 1-2-3-4 as above) Target covariance = Target_rho2

# (step 5) NEW PROBABILITIES [Moment-matching via Entropy Pooling]

# (step 1 [MomMatch routine]) Linear views
Aeq = r_[ones((1, t_)), epsi, epsi ** 2, epsi[[0]] * epsi[[1]]]

V1 = HFPmu
V2 = TargetCOV + HFPmu@HFPmu.T
beq = r_[array([[1]]),V1, array([[V2[0, 0]]]), array([[V2[1, 1]]]), array([[V2[0, 1]]])]

# (step 2 [MomMatch routine]) New Flex. Probs.
new_prob,_ = MinRelEntFP(flex_prob, None, None, Aeq, beq)
# -

# ## Figures

# +
# scatter colors (Twist-fix plot)
GreyRange = arange(0,0.83,10**-2)
CM, C = ColorCodedFP(flex_prob, 0, 0.0006, GreyRange, 0, 25, [25, 0])

for lag in [0,l]:
    f,ax = subplots(1,2, figsize=(12,6))

    # Twist fix for non-synchroneity
    plt.sca(ax[0])
    plt.axis('equal')
    if lag == 0:
        scatter(epsi[0], epsi[1], 5, c=C, marker='.',cmap=CM)
    else:
        scatter(new_epsi[0], new_epsi[1], 5, c=C, marker='.',cmap=CM)
    xlim([-0.08, 0.08])
    xticks(arange(-0.08,0.12,0.04))
    yticks(arange(-0.08,0.12,0.04))
    ylim([-0.08, 0.08])
    xlabel('S&P 500')
    ylabel('KOSPI')
    title('Twist fix')

    # Entropy Pooling fix for non-synchroneity in HFP
    # scatter colors

    plt.sca(ax[1])
    plt.axis('equal')
    if lag == 0:
        scatter(epsi[0], epsi[1], 5, c=C, marker='.',cmap=CM)
    else:
        [_, col1] = ColorCodedFP(new_prob, 0, 0.0006, arange(0,0.8,0.01), 0, 25, [22, 0])
        scatter(epsi[0], epsi[1], 5, c=col1, marker='.',cmap=CM)

    xlim([-0.08, 0.08])
    xticks(arange(-0.08,0.12,0.04))
    yticks(arange(-0.08,0.12,0.04))
    ylim([-0.08, 0.08])
    xlabel('S&P 500')
    ylabel('KOSPI')
    title('Entropy Pooling fix')
    plt.tight_layout()
    if lag == 0:
        Lag_string ='Overlap: 0 days'
    else:
        Lag_string = 'Overlap:  % 3.0f days'% l
    plt.text(0, 0.105, Lag_string);
    # save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
plt.show()
