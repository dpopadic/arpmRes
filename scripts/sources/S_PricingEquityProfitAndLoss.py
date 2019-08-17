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

# # S_PricingEquityProfitAndLoss [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_PricingEquityProfitAndLoss&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-4-equity-pl).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

from numpy import arange, ones, sort, where, diff, round, log, exp, sqrt
from numpy import sum as npsum, min as npmin, max as npmax

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, xlim, ylabel, \
    xlabel, title, xticks, yticks

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict
from FPmeancov import FPmeancov
from HistogramFP import HistogramFP
from SimulateBrownMot import SimulateBrownMot
# -

# ## Load the historical series of the S&P 500 from db_StocksS_P

try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_StocksS_P'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_StocksS_P'), squeeze_me=True)

# ## Select the last 1000 available observations of the daily price of the Priceline.com Inc equity (S&P 500 dataset with n = 279)
# ## and estimate the model parameters with Flexible Probabilities specified as rolling
# ## exponential decay prior where half-life is 1 year using function FPmeancov

# +
Data = struct_to_dict(db['Data'])

t_end = 240  # 240 days
dt = 0.5
horiz_u = arange(0,t_end+dt,dt)
prices = Data.Prices
index_stock = 279  # selected stock
x = log(prices[[index_stock-1], - 1000:])  # risk drivers (log-values) take the last 1000 observations
x_ = x.shape[1]
epsi = diff(x,1)  # invariants

lam = log(2) / 250  # half-life 1y
exp_decay = exp(-lam * (x_ - 1 - arange(0, x_-1, 1))).reshape(1,-1)
flex_probs_estimation = sort(exp_decay / npsum(exp_decay))  # sorted and normalized flexible probabilities
mu, var = FPmeancov(epsi, flex_probs_estimation)
sig = sqrt(var)
# -

# ## Simulate the risk driver as an arithmetic Brownian motion with drift using function SimulateBrownMot.

j_ = 9000
X = SimulateBrownMot(x[0,-1], horiz_u, mu, sig, j_)

# ## Compute the equity P&L, along with the mean and the standard deviation.

PL = exp(x[0,-1])*(exp(X - x[0,-1]) - 1)
Mu_PL = exp(x[0,-1]) * (exp((mu + 0.5*var)*horiz_u) - 1)
Sigma_PL = exp(x[0,-1]) * exp((mu + 0.5*var)*horiz_u) * sqrt(exp(horiz_u*var) - 1)

# ## Set the scenarios probabilities (equally weighted) and save the data in db_equity_PL.

flex_probs_scenarios = ones((1, j_)) / j_

# ## Plot z few simulated paths of the equity P&L up to 140 days,
# ## along with the expectation, the standard deviation and the horizon distribution.

# +
lgrey = [0.8, 0.8, 0.8]  # light grey
dgrey = [0.4, 0.4, 0.4]  # dark grey
j_sel = 15  # selected MC simulations
hor_sel = 140  # selected horizon for the plot (140 days)
i = where(horiz_u == hor_sel)[0][-1]

figure()
# simulated path, mean and standard deviation
plot(horiz_u[:i], PL[:j_sel,:i].T, color=lgrey)
plt.xticks(arange(0,t_end+20,20))
xlim([npmin(horiz_u), npmax(horiz_u)+1])
l1 = plot(horiz_u[:i], Mu_PL[0,:i], color='g', label='mean')
l2 = plot(horiz_u[:i], Mu_PL[0,:i] + Sigma_PL[0,:i], color='r', label=' + / - st.deviation')
plot(horiz_u[:i], Mu_PL[0,:i] - Sigma_PL[0,:i], color='r')

# histogram
option = namedtuple('option','n_bins')
option = namedtuple('option', 'n_bins')
option.n_bins = round(10*log(j_))
y_hist, x_hist = HistogramFP(PL[:, [i]].T, flex_probs_scenarios, option)
scale = 0.15 *Sigma_PL[0,i] / npmax(y_hist)
y_hist = y_hist*scale
shift_y_hist = horiz_u[i] + y_hist

# empirical pdf

emp_pdf = plt.barh(x_hist[:-1], shift_y_hist[0]-npmin(shift_y_hist[0]), left=npmin(shift_y_hist[0]), height=x_hist[1]-x_hist[0], facecolor=lgrey, edgecolor=lgrey, label='horizon pdf')
# border
plot(shift_y_hist[0], x_hist[:-1], color=dgrey)

legend()
xlabel('time (days)')
ylabel('P&L')
title('equity P&L');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
