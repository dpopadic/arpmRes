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

# # S_FlexibleCombinationsModels [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_FlexibleCombinationsModels&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerHeavyTails).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

from numpy import arange, ones, var, \
    percentile, round, mean, log, sqrt
from numpy import min as npmin, max as npmax

from scipy.stats import norm, t
from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, bar, legend, xlim, title

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict
from HistogramFP import HistogramFP
from Price2AdjustedPrice import Price2AdjustedPrice
from NormalMixtureFit import NormalMixtureFit
from CalibDegOfFreedomMLFP import CalibDegOfFreedomMLFP
# -

# ## Upload the database

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_Stocks'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_Stocks'), squeeze_me=True)

StocksSPX = struct_to_dict(db['StocksSPX'])
# -

# ## Compute the dividend-adjusted returns of one stock

_, epsi = Price2AdjustedPrice(StocksSPX.Date.reshape(1,-1), StocksSPX.Prices[[25],:], StocksSPX.Dividends[25])  # Cisco Systems Inc
t_ = epsi.shape[1]

# ## Empirical distribution fit
p = ones((1, t_)) / t_
option = namedtuple('option', 'n_bins')
option.n_bins = round(10*log(t_))
hgram, bins_epsi = HistogramFP(epsi, p, option)

# +
# ## Normal fit

epsi_grid = arange(npmin(epsi),npmax(epsi)+0.001,0.001)
mu = mean(epsi)
sigma2 = var(epsi)
normal = norm.pdf(epsi_grid, mu, sqrt(sigma2))
# -

# ## Gaussian mixture fit

# +
mu, Sigma, PComponents = NormalMixtureFit(epsi.T, 2, 0, 0, 0)
# fit = gmdistribution.fit(epsi.T,2)
# gauss_mixt = pdf(fit,epsi_grid.T)
gauss_mixt = PComponents[0,0]*norm.pdf(epsi_grid.reshape(-1,1), mu[0],sqrt(Sigma[0,0,0])) + PComponents[0,1]*norm.pdf(epsi_grid.reshape(-1,1),
                                                                                                   mu[1],
                                                                                                   sqrt(Sigma[0,0,1]))
# ## Student-t fit

p = ones((1, t_)) / t_  # historical probabilities

# the degrees of freedom are calibrated on the grid range(step_nu):max_nu
max_nu = 90
step_nu = 1

mu_t, sigma2_t, nu = CalibDegOfFreedomMLFP(epsi, p, max_nu, step_nu)
student = t.pdf((epsi_grid - mu_t) / sqrt(sigma2_t), nu) / sqrt(sigma2_t)
# -

# ## Create a figure showing the comparison between the estimated distributions

# +
q_inf = percentile(epsi, 100*0.0025)
q_sup = percentile(epsi, 100*0.9975)

# colors
blue = [0, 0, 0.4]
red = [0.9, 0.3, 0]
grey = [.9, .9, .9]
green = [.2, .6, .3]

f = figure()
p1 = bar(bins_epsi[:-1], hgram[0], width=bins_epsi[1]-bins_epsi[0], facecolor=[.9, .9, .9],edgecolor='k')
p2 = plot(epsi_grid, normal, color=green, lw=1.5)
p3 = plot(epsi_grid, gauss_mixt, color=red, lw=1.5)
p4 = plot(epsi_grid, student, color=blue, lw=1.5)
xlim([q_inf, q_sup])
leg = legend(handles=[p1[0],p2[0],p3[0],p4[0]],labels=['Empirical','Normal','Gaussian mixture', 'Student t(v=  %.1f)'%nu])
title('Heavy tails models');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
