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

# # S_MultiVarSqrRootRule [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_MultiVarSqrRootRule&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerSquareRootRuleVer).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import array, percentile, diff, cov, mean

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, legend, xlim, ylim, scatter, ylabel, \
    xlabel, title

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict
from intersect_matlab import intersect
from PlotTwoDimEllipsoid import PlotTwoDimEllipsoid
from RollPrices2YieldToMat import RollPrices2YieldToMat
# -

# ## Upload the database

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_SwapCurve'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_SwapCurve'), squeeze_me=True)

DF_Rolling = struct_to_dict(db['DF_Rolling'])
# -

# ## Compute yields, select observations and compute increments

# +
tau = [1, 5, 21]
nu = array([[2],[10]])  # times to maturity of interest (years)
y = {}

_, index, *_ = intersect(DF_Rolling.TimeToMat,nu)
# yields from rolling prices
y[0],_= RollPrices2YieldToMat(DF_Rolling.TimeToMat[index], DF_Rolling.Prices[index,:])  # yield daily observations
# extract weekly and monthly observations
for k in range(len(tau)):
    y[k] = y[0][:, ::tau[k]]  # computing increments
dy = {}
for k in range(3):
    dy[k] = diff(y[k], 1, 1)
# -

# ## Compute means and covariances

mu = {}
mu_tilde = {}
sigma2 = {}
sigma2_tilde = {}
for k in range(len(tau)):
    mu[k] = mean(dy[k], 1,keepdims=True)
    sigma2[k] = cov(dy[k], ddof=1)
    mu_tilde[k] = mu[0]*tau[k] / tau[0]  # projected daily mean
    sigma2_tilde[k] = sigma2[0]*tau[k] / tau[0]  # projected daily covariance

# ## Generate figures

# +
q_range=100*array([0.01, 0.99])
col =[0.94, 0.3, 0]

tit = {}
tit[0]= 'Daily observations'
tit[1]= 'Weekly observations'
tit[2]= 'Monthly observations'

for k in range(len(tau)):
    f=figure()
    scatter(dy[k][0], dy[k][1], 3, [.65, .65, .65], '*')
    xlim(percentile(dy[k][0], q_range))
    ylim(percentile(dy[k][1], q_range))
    xlabel('2 years yields increments')
    ylabel('10 years yields increments')

    h1 = PlotTwoDimEllipsoid(mu_tilde[k], sigma2_tilde[k], 1, 0, 0, 'g', 2)

    h2 = PlotTwoDimEllipsoid(mu[k], sigma2[k], 1, 0, 0, col, 2)

    if k > 0:
        h3 = PlotTwoDimEllipsoid(mu[0], sigma2[0], 1, 0, 0, [.6, .6, .6], 2)
        legend(handles=[h2[0][0], h3[0][0], h1[0][0]],labels=['empirical ellipsoid','daily ellipsoid','projected daily ellipsoid'])
    else:
        legend(handles=[h2[0][0]],labels=['empirical daily ellipsoid'])

    plt.grid(False)
    title(tit[k]);
    # save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
