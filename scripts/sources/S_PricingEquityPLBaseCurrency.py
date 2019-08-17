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

# # S_PricingEquityPLBaseCurrency [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_PricingEquityPLBaseCurrency&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-4-exch-equity-pl).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

from numpy import arange, ones, sort, where, diff, round, log, exp, sqrt, r_
from numpy import sum as npsum, max as npmax

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, xlim, ylabel, \
    xlabel, title, xticks

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict
from FPmeancov import FPmeancov
from intersect_matlab import intersect
from HistogramFP import HistogramFP
from SimulateBrownMot import SimulateBrownMot
# -

# ## Load the historical series of the S&P 500 from StocksS_P
# ## and the historical series of hte daily exchange rate from db_FX.

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_StocksS_P'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_StocksS_P'), squeeze_me=True)

Data = struct_to_dict(db['Data'])

try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_FX'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_FX'), squeeze_me=True)

USD_GBP = struct_to_dict(db['USD_GBP'])
# -

# ## Select the daily price of the Priceline.com Inc equity (S&P 500 dataset with n= 279)
# ## and the USD/GBP daily exchange rate (USD_GBP.FX from db_FX), and compute the risk drivers,
# ## that are the log-value for the equity and the log-rate for the spot exchange rate.

# +
t_end = 240
dt = 0.5
horiz_u = arange(0,t_end+dt,dt)
dates_stock = Data.Dates
dates_fx = USD_GBP.Date
[dates, i_stock, i_fx] = intersect(dates_stock, dates_fx)  # match the db

# risk drivers and invariants for the stock price
index_stock = 278  # choose the stock
x = log(Data.Prices[index_stock, i_stock])
t_ = len(x)
epsi_stock = diff(x)

# risk drivers and invariants for the foreign exchange rate
fx_USD_GBP = log(USD_GBP.FX[i_fx])
epsi_fx = diff(fx_USD_GBP)
# -

# ## Estimate the input parameters with Flexible Probabilities specified as rolling exponential decay prior
# ## where half-life is 1 year using function FPmeancov.

# +
lam = log(2) / 250  # half-life 1y
exp_decay = exp(-lam*(t_ - 1 - arange(0, t_ - 2 + 1, 1))).reshape(1,-1)
flex_probs_estimation = sort(exp_decay / npsum(exp_decay))  # sorted and normalized flexible probabilities
mu, var = FPmeancov(r_[epsi_stock.reshape(1,-1), epsi_fx.reshape(1,-1)], flex_probs_estimation)

mu_stock = mu[0]
sig_stock = sqrt(var[0, 0])
mu_fx = mu[1]
sig_fx = sqrt(var[1, 1])
mu = mu.T@ones((len(mu), 1))
sig = sqrt(ones((1, len(var)))@var@ones((len(var), 1)))
# -

# ## Simulate the log-value risk driver and the log-rate risk driver as a bivariate Brownian motion using function SimulateBrownMot
# ## and compute the mean and the standard deviation of the foreign exchange rate.

j_ = 3000
X = SimulateBrownMot(x[-1], horiz_u, mu_stock, sig_stock, j_)
Z = SimulateBrownMot(fx_USD_GBP[-1], horiz_u, mu_fx, sig_fx, j_)
FX = exp(Z)
Mu_FX = exp((fx_USD_GBP[-1])) * exp((mu_fx + 0.5*sig_fx ** 2)*horiz_u)
Sigma_FX = exp((fx_USD_GBP[-1])) * exp((mu_fx + 0.5*sig_fx ** 2)*horiz_u) * sqrt(exp(horiz_u*sig_fx ** 2) - 1)

# ## Compute the equity P&L in local currency ($) with the corresponding mean and standard deviation,
# ## the equity P&L in base currency (pound) along with the mean and the standard deviation,
# ## and the local currency P&L normalized to base currency with the corresponding mean and standard deviation

# +
PL_l = exp((x[-1])) * (exp(X - x[-1]) - 1)  # P&L in local currency (dollar)
Mu_PL_l = exp(x[-1]) * (exp((mu_stock + 0.5*sig_stock ** 2)*horiz_u) - 1)
Sigma_PL_l = exp(x[-1]) * exp((mu_stock + 0.5*sig_stock ** 2)*horiz_u) * sqrt(exp(horiz_u*sig_stock ** 2) - 1)

PL_b = FX * PL_l  # P&L in base currency (pound)
Mu_PL_b = exp(x[-1]) * exp(fx_USD_GBP[-1])*(exp((mu + 0.5*sig ** 2)*horiz_u) - 1)
Sigma_PL_b = exp(x[-1]) * exp(fx_USD_GBP[-1])*exp((mu + 0.5*sig ** 2)*horiz_u)*sqrt(exp(horiz_u*(sig ** 2)) - 1)

PL_norm = PL_l * exp(fx_USD_GBP[-1])  # local currency P&L normalized to base currency for comparison (pound)
Mu_PL_norm = Mu_PL_l * exp(fx_USD_GBP[-1])  # P&L local currency mean normalized to base currency for comparison (pound)
Sigma_PL_norm = Sigma_PL_l * exp(fx_USD_GBP[-1])  # P&L local currency std. normalized to base currency for comparison (pound)
# -

# ## Set the scenarios probabilities (equally weighted).

flex_probs_scenarios = ones((j_, 1)) / j_

# ## Plot few (say 15) simulated paths of the foreign exchange rate up to 140 days,
# ## along with the expectation, the standard deviation and the horizon distribution.
# ## Furthermore, plot few (say 15) simulated paths of the equity P&L in base currency (pound),
# ## along with the mean, the standard deviation and the horizon distribution,
# ## and also the mean, the standard deviation and the horizon distribution of
# ## the equity local currency P&L normalized to base currency.

# +
lgrey = [0.8, 0.8, 0.8]  # light grey
dgrey = [0.4, 0.4, 0.4]  # dark grey
j_sel = 15  # selected MC simulations
hor_sel = 140  # selected horizon for the plot (140 days)
i = where(horiz_u == hor_sel)[0][0]

# foreign exchange rate figure
figure()

# simulated path, mean and standard deviation
plot(horiz_u[:i+1], FX[:j_sel, :i+1].T, color=lgrey, lw=1)
xticks(arange(0,t_end+1,20))
xlim([min(horiz_u), max(horiz_u)+1])
l1 = plot(horiz_u[:i+1], Mu_FX[:i+1], color='g')
l2 = plot(horiz_u[:i+1], Mu_FX[:i+1] + Sigma_FX[:i+1], color='r')
plot(horiz_u[:i+1], Mu_FX[:i+1] - Sigma_FX[:i+1], color='r')
# histogram
option = namedtuple('option', 'n_bins')
option.n_bins = round(10*log(j_))
y_hist, x_hist = HistogramFP(FX[:,[i]].T, flex_probs_scenarios.T, option)
scale = 1500*Sigma_FX[i] / npmax(y_hist)
y_hist = y_hist*scale
shift_y_hist = horiz_u[i] + y_hist
# empirical pdf

emp_pdf = plt.barh(x_hist[:-1], shift_y_hist[0]-horiz_u[i], height=x_hist[1]-x_hist[0], left=horiz_u[i],
                   facecolor=lgrey, edgecolor= lgrey)
plot(shift_y_hist[0], x_hist[:-1], color=dgrey, lw=1)  # border

legend(handles=[l1[0],l2[0],emp_pdf[0]],labels=['mean',' + / - st.deviation','horizon pdf'])
xlabel('time (days)')
ylabel('USD / GBP')
title('Foreign exchange rate USD/GBP')

# P&L in base currency (pound) figure
figure()
# simulated path, mean and standard deviation
plot(horiz_u[:i+1], PL_b[:j_sel, :i+1].T, color=lgrey, lw=1)
xticks(arange(0,t_end,20))
xlim([min(horiz_u), max(horiz_u)+1])
l1 = plot(horiz_u[:i+1], Mu_PL_b[0,:i+1], color='g')
l2 = plot(horiz_u[:i+1], Mu_PL_b[0,:i+1] + Sigma_PL_b[0,:i+1], color='r')
plot(horiz_u[:i+1], Mu_PL_b[0,:i+1] - Sigma_PL_b[0,:i+1], color='r')
# normalized P&L
l3 = plot(horiz_u[:i+1], Mu_PL_norm[:i+1], linestyle='--',color='k')
plot(horiz_u[:i+1], Mu_PL_norm[:i+1] + Sigma_PL_b[0,:i+1], linestyle='--',color='k')
plot(horiz_u[:i+1], Mu_PL_norm[:i+1] - Sigma_PL_b[0,:i+1], linestyle='--',color='k')
# histogram
y_hist, x_hist = HistogramFP(PL_b[:,[i]].T, flex_probs_scenarios.T, option)
scale2 = 0.4*Sigma_PL_b[0,i] / npmax(y_hist)
y_hist = y_hist*scale2
shift_y_hist = horiz_u[i] + y_hist
# empirical pdf

emp_pdf = plt.barh(x_hist[:-1], shift_y_hist[0]-horiz_u[i], left=horiz_u[i], height=x_hist[1]-x_hist[0],facecolor=lgrey, edgecolor= lgrey)
plot(shift_y_hist[0], x_hist[:-1], color=dgrey, lw=1)  # border

# histogram
y_hist2, x_hist2 = HistogramFP(PL_norm[:,[i]].T, flex_probs_scenarios.T, option)
y_hist2 = y_hist2*scale2
shift_y_hist2 = horiz_u[i] + y_hist2
plot(shift_y_hist2[0], x_hist2[:-1], color=dgrey,linestyle='--') # border

legend(handles=[l1[0], l2[0], emp_pdf[0], l3[0]],labels=['P&L base currency (GBP) mean','P & L base currency(GBP) + / - st.deviation',
        'P&L base currency (GBP) horizon pdf','P&L local currency (USD) normalized features'])
xlabel('time (days)')
ylabel('Equity P&L')
title('Equity P&L');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
