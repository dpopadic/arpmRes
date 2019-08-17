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

# # S_ProjectionStudent [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_ProjectionStudent&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=EXStudtlProjection).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import arange, array, ones, cumsum, abs, log, exp, sqrt, r_
import numpy as np
from numpy import sum as npsum

from scipy.stats import norm
from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, ylabel, \
    xlabel, title

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import struct_to_dict, save_plot
from Price2AdjustedPrice import Price2AdjustedPrice
from ProjDFFT import ProjDFFT
from CalibDegOfFreedomMLFP import CalibDegOfFreedomMLFP
# -

# ## Upload database db_Stocks

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_Stocks'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_Stocks'), squeeze_me=True)

StocksSPX = struct_to_dict(db['StocksSPX'])
# -

# ## Compute the log-value and the log-returns time series from the data referring to CISCO Systems Inc

# +
index = 25  # Cisco Systems Inc

[_, dx] = Price2AdjustedPrice(StocksSPX.Date.reshape(1,-1), StocksSPX.Prices[[index], :], StocksSPX.Dividends[index])  # dividend-adjusted log-returns
x = cumsum(dx)  # dividend-adjusted log-values
t_ = len(x)
# -

# ## Compute the Flexible Probabilities

#exponential decay
lam = log(2) / 800  # half life 3y
p = exp(-lam*abs(arange(t_, 1 + -1, -1))).reshape(1,-1)
p = p /npsum(p)

# ## Fit the Student t distribution by using function CalibDegOfFreedomMLFP

mu, sigma2, nu = CalibDegOfFreedomMLFP(dx, p, 10, 0.1)

# ## Project the expectation and standard deviation

# +
tau = 20  # horizon
k_ = 2 ** 12

mu_tau = x[t_-1] + mu*tau
sigma_tau = sqrt(sigma2*tau*nu / (nu - 2))
# -

# ## Project the estimated pdf to the horizon via the FFT algorithm

x_hat_hor, f_hat_hor,*_ = ProjDFFT(None, None, tau, k_, 'Student t', nu)
x_hor = mu*tau*ones((1, len(x_hat_hor))) + sqrt(sigma2)*x_hat_hor
f_hor = f_hat_hor / sqrt(sigma2)
f_hor = np.real(f_hor)

# ## Compute the normal approximation of the projected pdf

phi_hor = norm.pdf(x_hor, mu*tau, sigma_tau)
# center around x[t_end-1]
x_hor = x_hor + x[t_-1]

# ## Create a figure

# +
s_ = 2  # number of plotted observation before projecting time

# axes settings
m = min([min(x[t_ - s_:t_]), mu_tau - 5*sigma_tau])
M = max([max(x[t_ - s_:t_]), mu_tau + 5*sigma_tau])
t = arange(-s_,tau+1)
max_scale = tau / 4

# preliminary computations
tau_red = arange(0,tau+0.1,0.1)
mu_red = x[t_-1] + mu*tau_red
sigma_red = sqrt(sigma2*nu / (nu - 2))*sqrt(tau_red)
redline1 = mu_red + 2*sigma_red
redline2 = mu_red - 2*sigma_red

f = figure()
# color settings
lgrey = [0.8, 0.8, 0.8]  # light grey
dgrey = [0.2, 0.2, 0.2]  # dark grey
lblue = [0.27, 0.4, 0.9]  # light blue
plt.axis([t[0], t[-1] + max_scale, m, M])
xlabel('time (days)')
ylabel('Risk driver')

plt.grid(False)
title('Student T projection')
# standard deviation lines
p_red_1 = plot(tau_red, redline1, color='r', lw = 2)  # red bars (+2 std dev)
p_red_2 = plot(tau_red, redline2, color='r', lw = 2)  # red bars (-2std dev)
p_mu = plot([0, tau], [x[t_-1], mu_tau], color='g', lw = 2)  # expectation
# histogram pdf plot
plot(r_['-1',tau*ones((f_hor.shape[0],1)), tau+f_hor.reshape(-1,1)].T, r_['-1',x_hor[[0]].T, x_hor[[0]].T].T, color=lgrey, lw=2)
f_border = plot(tau+f_hor, x_hor[0], color=dgrey, lw=1)
# normal approximation plot
phi_border = plot(tau+phi_hor[0], x_hor[0], color=lblue, lw=1)
# plot of last s_ observations
for k in range(s_):
    plot([t[k], t[k+1]], [x[t_-s_+k-1], x[t_-s_+k]], color=lgrey, lw=2)
    plot(t[k], x[t_-s_+k-1], color='b',linestyle='none', marker='.',markersize=15)
plot(t[s_], x[t_-1], color='b',linestyle='none', marker='.',markersize=15)
plot(tau, mu_tau -2*sigma_tau, color='r', lw = 2)
plot(tau, 2*sigma_tau, color='r', lw = 2)
legend(handles=[f_border[0], phi_border[0], p_mu[0], p_red_1[0]],labels=['horizon pdf','normal approximation','expectation',' + / - 2st.deviation']);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
