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

# # S_ProjectionFractionalBM [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_ProjectionFractionalBM&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=FracBMProj).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import arange, ones, zeros, cumsum, diff, linspace, sqrt, tile, r_
from numpy import min as npmin, max as npmax

from scipy.stats import norm
from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, ylabel, \
    xlabel, title

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot
from FPmeancov import FPmeancov
from FitFractionalIntegration import FitFractionalIntegration
from ffgn import ffgn
# -

# ## Upload database

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_SwapParRates'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_SwapParRates'), squeeze_me=True)

Rates = db['Rates']
Dates = db['Dates']
# -

# ## Compute swap par rates increments

x = Rates[0]  # select time series corresponding to 1y par rates
dates = Dates
dx = diff(x)

# ## Fit the parameters of fractional Brownian motion

# +
lags = 50
d0 = 0

d = FitFractionalIntegration(dx, lags, d0)[0]
h = d + 0.5  # Hurst coefficient

t_ = len(dx)
[mu, sigma2] = FPmeancov(dx.reshape(1,-1), ones((1, t_)) / t_)
# -

# ## Initialize projection variables

tau = 252  # investment horizon of 1 year (expressed in days)
dt = 1  # infinitesimal step for simulations
t_j = arange(0,tau+dt,dt)
j_ = 15  # number of simulated paths

# ## Simulate paths
h = 0.091924700639547 + 0.5

# +
dW = ffgn(h, j_, len(t_j) - 1)
W = r_['-1',zeros((j_, 1)), cumsum(dW, 1)]
mu_t = mu*t_j

X = tile(mu_t, (j_, 1)) + sqrt(sigma2*dt**(2*h))*W
X = x[-1] + X
# -

# ## Projection to horizon

# +
# moments
mu_tau = x[-1] + mu.squeeze()*tau
sigma_tau = sqrt(sigma2.squeeze()*tau ** (2*h))
sigma_norm = sqrt(sigma2.squeeze()*tau)

# analytical pdf
l_ = 2000
x_hor = linspace(mu_tau-4*sigma_tau,mu_tau+4*sigma_tau, l_)
y_hor = norm.pdf(x_hor, mu_tau, sigma_tau)
# -

# ## Generate figure

# +
s_ = 42  # number of plotted observation before projecting time

# axes settings
m = min([npmin(x[- s_:]), x[-1]-4*sigma_tau, npmin(X)])
M = max([npmax(x[- s_:]), x[-1] + 4*sigma_tau, npmax(X)])
t = arange(-s_,tau+1)
max_scale = tau / 4
scale = max_scale / npmax(y_hor)

# preliminary computations
mu_red = x[-1] + mu_t
sigma_red = sqrt(sigma2*t_j**(2*h))
redline1 = mu_red + 2*sigma_red
redline2 = mu_red - 2*sigma_red
sigma_blue = sqrt(sigma2*t_j)
blueline1 = mu_red + 2*sigma_blue
blueline2 = mu_red - 2*sigma_blue

f = figure()

# color settings
lgrey = [0.8, 0.8, 0.8]  # light grey
dgrey = [0.2, 0.2, 0.2]  # dark grey
lblue = [0.27, 0.4, 0.9]  # light blue

plt.axis([t[0], t[-1] + max_scale, m, M])
xlabel('time (days)')
ylabel('Risk driver')
plt.grid(False)
title('Fractional Brownian motion')

# simulated paths
for j in range(j_):
    plot(t_j, X[j,:].T, color = lgrey, lw = 2)

# standard deviation lines
p_blue_1 = plot(t_j, blueline1[0], color='b', lw = 2)  # red bars (+2 std dev)
p_blue_2 = plot(t_j, blueline2[0], color='b', lw = 2)  # red bars (-2std dev)
p_red_1 = plot(t_j, redline1[0], color='r', lw = 2)  # red bars (+2 std dev)
p_red_2 = plot(t_j, redline2[0], color='r', lw = 2)  # red bars (-2std dev)
p_mu = plot([0, tau], [x[-1], mu_tau], color='g', lw = 2)  # expectation

# histogram pdf plot
for k in range(len(y_hor)):
    plot([tau, tau+y_hor[k]*scale], [x_hor[k], x_hor[k]], color=lgrey, lw=2)

f_border = plot(tau+y_hor*scale, x_hor, color=dgrey, lw=1)

# plot of last s_ observations
for k in range(s_):
    plot([t[k], t[k+1]], [x[-s_+k-1], x[-s_+k]], color=lgrey, lw=2)
    plot(t[k], x[-s_+k-1], color='b',linestyle='none',marker= '*', markersize=3)

plot(t[s_], x[-1], color='b',linestyle='none',marker= '.', markersize=3)

# leg
#
legend(handles=[p_mu[0], p_red_1[0], f_border[0], p_blue_1[0]],labels=['expectation','+ / - 2 st.deviation', 'horizon pdf','+ / -2 st.dev Brownian motion']);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

