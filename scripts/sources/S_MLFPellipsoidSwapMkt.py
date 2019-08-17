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

# # S_MLFPellipsoidSwapMkt [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_MLFPellipsoidSwapMkt&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=exer-mfpellipt-copy-3).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import arange, array, zeros, percentile, diff, log, exp
from numpy import sum as npsum, min as npmin, max as npmax

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, xlim, ylim, scatter, ylabel, \
    xlabel, title

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict
from PlotTwoDimEllipsoid import PlotTwoDimEllipsoid
from RollPrices2YieldToMat import RollPrices2YieldToMat
from MaxLikelihoodFPLocDispT import MaxLikelihoodFPLocDispT
from ColorCodedFP import ColorCodedFP
# -

# ## Upload database

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_SwapCurve'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_SwapCurve'), squeeze_me=True)

DF_Rolling = struct_to_dict(db['DF_Rolling'])
# -

# ## Select the last 1000 yield observations with 2 and 5 years to maturity

# +
t_ = 1000
tau = [2,5]

y = zeros((2,t_))

# yields from rolling pricing
y[0,:],_ = RollPrices2YieldToMat(tau[0],DF_Rolling.Prices[DF_Rolling.TimeToMat == tau[0], - t_ :])  # 2yrs yields
y[1,:],_ = RollPrices2YieldToMat(tau[1],DF_Rolling.Prices[DF_Rolling.TimeToMat == tau[1], - t_ :])  # 5yrs yields
# -

# ## Compute the invariants

epsi = diff(y, 1, 1)  # rate daily changes

# ## Maximum Likelihood with Flexible Probabilities (MLFP) Student t fit

# +
# degrees of freedom
nu = 5

# flexible probabilities (exponential decay half life 6 months)
lam = log(2) / 180
p = exp(-lam*arange(t_ - 1, 1 + -1, -1)).reshape(1,-1)
p = p /npsum(p)

# Fit
tolerance = 10 ** (-10)
mu_MLFP, sigma2_MLFP,_ = MaxLikelihoodFPLocDispT(epsi, p, nu, tolerance, 1)

# Student t mean and covariance
m_MLFP = mu_MLFP
s2_MLFP = nu / (nu - 2)*sigma2_MLFP
# -

# ## Create figures

# +
CM, C = ColorCodedFP(p, npmin(p), npmax(p), arange(0,0.8,0.005), 0, 1, [1, 0])

f = figure()
# colormap(CM)
scatter(epsi[0], epsi[1], 10, c=C, marker='.', cmap=CM) #color-coded scatter plot

PlotTwoDimEllipsoid(m_MLFP.reshape(-1,1), s2_MLFP, 1, 0, 0, [.9, .4, 0])  # MLFP ellipsoid
xlim(percentile(epsi[0], 100*array([0.01, 0.99])))
ylim(percentile(epsi[1], 100*array([0.01, 0.99])))
xlabel('2 yr swap rate daily changes')
ylabel('5 yr swap rate daily changes')
title('MLFP-ellipsoid');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
