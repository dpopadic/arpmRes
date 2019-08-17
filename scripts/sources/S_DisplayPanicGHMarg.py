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

# # S_DisplayPanicGHMarg [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_DisplayPanicGHMarg&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-2-pani-cop-ghmarg).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import arange, zeros, percentile, diff, round, log, exp, corrcoef
from numpy import sum as npsum

from scipy.stats import norm
from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, bar, scatter, ylabel, \
    xlabel, title

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict
from intersect_matlab import intersect
from HistogramFP import HistogramFP
from ConditionalFP import ConditionalFP
from PanicTDistribution import PanicTDistribution
from CopMargSep import CopMargSep
from ColorCodedFP import ColorCodedFP
from GHCalibration import GHCalibration

# inputs
j_ = 1000  # number of simulations
nb = round(5*log(j_))

nu = 3  # degree of freedom
r = 0.95  # panic correlation
c = 0.07  # threshold

# Load daily observations of the stocks in S&P 500
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_StocksS_P'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_StocksS_P'), squeeze_me=True)

Data = struct_to_dict(db['Data'])

V = Data.Prices
pair = [0, 1]  # stocks to spot

# Set the calm correlation matrix as sample correlation matrix of compounded returns
C = diff(log(V), 1, 1)
C = C[pair, :]
varrho2 = corrcoef(C)

# Compute panic distribution
X, p_ = PanicTDistribution(varrho2, r, c, nu, j_)

# Extract the simulations of the panic copula
x, u, U = CopMargSep(X, p_)
# -

# ## Load the observations of VIX

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_VIX'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_VIX'), squeeze_me=True)

VIX = struct_to_dict(db['VIX'])

Z = VIX.value
Vdates = VIX.Date
dates_Stocks = Data.Dates

# match the db
Dates, i_c, i_vix = intersect(dates_Stocks[1:], Vdates)
C = C[:, i_c]
Z_VIX = Z[i_vix]

n_, t_ = C.shape
# -

# ## Compute Historical distribution with Flexible Probabilities conditioned on the VIX

# +
lam = 0.0005

# exponential decay Flexible Probabilities (prior)
prior = zeros((1, t_))
for t in range(t_):
    prior[0,t] = exp(-(t_ - t)*lam)

prior = prior / npsum(prior)
VIX = namedtuple('VIX', 'Series TargetValue Leeway')
VIX.Series = Z_VIX.reshape(1,-1)
VIX.TargetValue = np.atleast_2d(percentile(Z_VIX, 100 * 0.7))
VIX.Leeway = 0.3

p = ConditionalFP(VIX, prior)  # FP conditioned on the VIX
# -

# ## Fit the g&h inverse cdf to the Historical quantiles via Flexible Probabilities

# +
# step of local search
Da0 = 1.0e-4
Db0 = 1.0e-4
Dg0 = 1.0e-4
Dh0 = 1.0e-4

Tolerance = 1.0e-8
MaxItex = 10000  # maximun number of iterations

aGH, bGH, gGH, hGH, SqDistGH, iterGH = GHCalibration(C, p, Tolerance, Da0, Db0, Dg0, Dh0, MaxItex)
# -

# ## Compute the simulations of the g&h Marginals by feeding the panic copula to the g&h inverse cdf

Y = zeros((n_, j_))
for n in range(n_):
    Y[n, :] = aGH[n] + bGH[n]*((1 / gGH[n])*(exp(gGH[n]*norm.ppf(U[n, :], 0, 1)) - 1)
                               *exp(0.5*hGH[n]*norm.ppf(U[n, :], 0, 1) ** 2))

# ## Represent the scatter-plot and plot the histograms of the g&h marginals

# +
# scatter plot
figure()
grey_range = arange(0,0.81,0.01)
CM, C = ColorCodedFP(p_, None, None, grey_range, 0, 18, [17, 5])
# colormap(CM)
scatter(Y[0], Y[1], s=3, c=C, marker='.',cmap=CM)
xlabel('$Y_1$')
ylabel('$Y_2$')
title('g&h distribution');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

# g&h marginal Y1
figure()
option = namedtuple('option', 'n_bins')
option.n_bins = nb
n1, c1 = HistogramFP(Y[[0]], p_, option)
bar(c1[:-1], n1[0], width=c1[1]-c1[0], facecolor=[.9, .9, .9], edgecolor=  'k')
title('Marginal $Y_1$');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

# g&h marginal Y2
figure()
n2, varrho2 = HistogramFP(Y[[1]], p_, option)
bar(varrho2[:-1], n2[0], width=varrho2[1]-varrho2[0], facecolor=[.9, .9, .9], edgecolor=  'k')
title('Marginal $Y_2$');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

