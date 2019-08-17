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

# # S_PricingOptionsHFP [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_PricingOptionsHFP&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-call-option-value-hist).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import arange, reshape, ones, zeros, diag, round, log, exp, sqrt, tile, r_, newaxis, real

from scipy.io import loadmat, savemat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, xlim, scatter, ylabel, \
    xlabel, title, xticks

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, interpne
from FPmeancov import FPmeancov
from HistogramFP import HistogramFP
from ColorCodedFP import ColorCodedFP
from PerpetualAmericanCall import PerpetualAmericanCall
from blsprice import blsprice
# -

# ## Upload the database db_ProjOptionsHFP (computed in S_ProjectionOptionHFP) and compute the projected
# ## underlying values and the projected short rates

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_ProjOptionsHFP'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_ProjOptionsHFP'), squeeze_me=True)

x_1 = real(db['x_1'].reshape(1, -1))
x_1hor = real(db['x_1hor'])
x_2 = real(db['x_2'].reshape(1, -1))
x_2hor = real(db['x_2hor'])
x_3 = real(db['x_3'])
x_3hor = real(db['x_3hor'])
tau = db['tau']
j_ = db['j_']
n_ = db['n_']
p = db['p'].reshape(1,-1)
eta = db['eta']
n_grid = db['n_grid']
sigma_m = db['sigma_m']
m_grid = db['m_grid']
maturity = db['maturity']

X_1 = r_['-1',tile(x_1[0,[-1]], (j_, 1)), x_1hor]  # projected log-underlying values
X_2 = r_['-1',tile(x_2[0,[-1]], (j_, 1)), x_2hor]  # projected shadow short rates
X_3 = r_['-1',tile(x_3[:,[-1]], (1, j_))[...,newaxis], x_3hor]
X_3 = reshape(X_3, (n_, n_grid + 1, j_, tau + 1),'F')  # projected log-implied volatilities
V = exp(X_1)  # projected underlying values
Y = PerpetualAmericanCall(X_2, {'eta':eta})  # projected short rates

# parameters
k = exp(x_1[0,-1])  # at the money strike
tau_options = 10 / 252  # options time to maturity
# -

# ## Compute the call and the put option value at the current time

moneyness_tnow = log(exp(x_1[0,-1]) / k) / sqrt(tau_options)  # moneyness of the options at time t_now
sigmatmp = sigma_m[:,:, -1]
sigma_tnow = interpne(sigmatmp, r_[tau_options, moneyness_tnow], [maturity, m_grid])
vcall_tnow, vput_tnow = blsprice(exp(x_1[0,-1]), k, Y[0, 0], tau_options, sigma_tnow),\
                        blsprice(exp(x_1[0,-1]), k, Y[0, 0], tau_options, sigma_tnow,cp=-1)  # BS call and put option values at time t_now

# ## Compute the call and the put option values at the horizons t_hor = t_now + tau, where tau = 1,...,6

# +
Vcall = zeros((j_, tau))
Vput = zeros((j_, tau))
Moneyness = zeros((j_, tau))
LogSigma = zeros((j_, tau))

for t in range(tau):
    tau_hor = tau_options-t / 252  # time to maturity of the options at the horizon
    Moneyness[:,t] = log(V[:,t] / k) / sqrt(tau_hor)  # moneyness of the options at the horizon

    # interpolated log-implied volatility of the options at the horizon
    for j in range(j_):
        LogSigma[j, t] = interpne(X_3[:,:, j, t], r_[tau_hor, Moneyness[j, t]], [maturity, m_grid])

    Vcall[:,t], Vput[:,t] = blsprice(V[:,t], k, Y[:,t], tau_hor, exp(LogSigma[:,t])), blsprice(V[:,t], k, Y[:,t], tau_hor, exp(LogSigma[:,t]), cp=-1)
# -

# ## Compute the call and put option P&L scenarios at the horizons, together with their mean and standard deviation

# +
# call option P&L scenarios
Pi_call = Vcall - vcall_tnow * ones(Vcall.shape)
[MuPi_call, SigmaPi_call] = FPmeancov(Pi_call.T, p)
SigmaPi_call = sqrt(diag(SigmaPi_call))

# put option P&L scenarios
Pi_put = Vput - vput_tnow * ones(Vput.shape)
[MuPi_put, SigmaPi_put] = FPmeancov(Pi_put.T, p)
SigmaPi_put = sqrt(diag(SigmaPi_put))

# portfolio P&L scenarios
Pi_ptf = r_[Pi_call, Pi_put]
# -

# ## Save the data in PricOptionsHFP
vars_to_save = {varname: var for varname, var in locals().items() if isinstance(var, (np.ndarray,np.int,np.float))}
savemat(os.path.join(TEMPORARY_DB, 'db_PtocOptionsHFP'), vars_to_save)

# +
# ## Plot a few simulated paths (say 15) of the call and put option P&L at the selected horizon (t_hor = t_now + 6 days),
# ## along with the expectation, the standard deviation and the horizon distribution.
# ## Furthermore, show the scatter plot of the call and put option P&L as a function of the underlying value.

GreyRange = arange(0,0.86,0.01)
CM, C = ColorCodedFP(p, None, None, GreyRange, 0, 22, [18, 7])

lgrey = [0.8, 0.8, 0.8]  # light grey
dgrey = [0.4, 0.4, 0.4]  # dark grey
j_sel = 15  # selected number of printed paths

# call option P&L distribution
figure()
# simulated path, mean and standard deviation
plot(arange(1,tau+1), Pi_call[:j_sel+1,:].T, color = lgrey, lw=1)
xticks(arange(1,tau+1))
xlim([1, tau + 10])
l1 = plot(arange(1,tau+1), MuPi_call.flatten(), color='g')
l2 = plot(arange(1,tau+1), MuPi_call.flatten() + SigmaPi_call.flatten(), color='r')
plot(arange(1,tau+1), MuPi_call.flatten() - SigmaPi_call.flatten(), color='r')
# histogram
option = namedtuple('option', 'n_bins')
option.n_bins = round(10*log(j_))
y_hist, x_hist = HistogramFP(Pi_call[:,[-1]].T, p, option)
scale = 10 / max(y_hist[0])
y_hist = y_hist*scale
shift_y_hist = tau + y_hist
# empirical pdf
emp_pdf = plt.barh(x_hist[:-1], shift_y_hist[0]-tau, left=tau, height=x_hist[1]-x_hist[0],facecolor=lgrey, edgecolor= lgrey, lw=2)
legend(handles=[l1[0],l2[0],emp_pdf], labels=['mean','+ / - st.deviation','horizon pdf'])
xlabel('time (years)')
ylabel('Call option P&L')
title('Call option projected P&L at the horizon');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

# scatter plot call option P&L
figure()
# colormap(CM)
plt.gca().set_facecolor('white')
scatter(V[:,-1], Pi_call[:,-1], 10, c=C, marker='.',cmap=CM)
xlabel('Underlying')
ylabel('Call option P&L')
title('Scatter plot call option P&L vs. underlying');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

# put option P&L distribution
figure()
# simulated path, mean and standard deviation
plot(arange(1,tau+1), Pi_put[:j_sel+1,:].T, color = lgrey, lw=1)
xticks(arange(1,tau+1)), xlim([1, tau + 10])
l1 = plot(arange(1,tau+1), MuPi_put.flatten(), color='g')
l2 = plot(arange(1,tau+1), MuPi_put.flatten() + SigmaPi_put.flatten(), color='r')
plot(arange(1,tau+1), MuPi_put.flatten() - SigmaPi_put.flatten(), color='r')
# histogram
option = namedtuple('option', 'n_bins')
option.n_bins = round(10*log(j_))
y_hist, x_hist = HistogramFP(Pi_put[:,[-1]].T, p, option)
scale = 10 / max(y_hist[0])
y_hist = y_hist*scale
shift_y_hist = tau + y_hist
# empirical pdf
emp_pdf = plt.barh(x_hist[:-1], shift_y_hist[0]-tau, left=tau, height=x_hist[1]-x_hist[0], facecolor=lgrey, edgecolor= lgrey, lw=2)
legend(handles=[l1[0],l2[0],emp_pdf], labels=['mean',' + / - st.deviation','horizon pdf'])
xlabel('time (years)')
ylabel('Put option P&L')
title('Put option projected P&L at the horizon');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

# scatter plot call option P&L
figure()
# colormap(CM)
plt.gca().set_facecolor('white')
scatter(V[:,-1], Pi_put[:,-1], 10, c=C, marker='.',cmap=CM)
xlabel('Underlying')
ylabel('Put option P&L')
title('Scatter plot put option P&L vs. underlying');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
