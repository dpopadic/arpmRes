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

# # S_ProjectionPoisson [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_ProjectionPoisson&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerPoissProcProj).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

from numpy import arange, array, ceil, linspace, log, exp, sqrt, unique
from numpy import sum as npsum, min as npmin, max as npmax

from scipy.stats import norm, poisson
from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, ylabel, \
    xlabel, title

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict
from EffectiveScenarios import EffectiveScenarios
from JumpDiffusionMerton import JumpDiffusionMerton
from binningHFseries import binningHFseries
from IterGenMetMomFP import IterGenMetMomFP
# -

# ## Upload databases

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_US_10yr_Future_quotes_and_trades'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_US_10yr_Future_quotes_and_trades'), squeeze_me=True)

trades = struct_to_dict(db['trades'])
# -

# ## Compute the invariants

trade_time = unique(trades.time)
flag = '1second'
epsi, k,*_ = binningHFseries(trade_time, flag)
k_ = len(k)

# ## Set the Flexible Probabilities

lam = log(2) / 360
p = exp((-lam * arange(k_, 1 + -1, -1))).reshape(1,-1)
p = p /npsum(p)  # FP-profile: exponential decay 1 years
typ = namedtuple('type','Entropy')
typ.Entropy = 'Exp'
ens = EffectiveScenarios(p, typ)

# ## Estimation of the 1-step pdf with the Generalized Method of Moments

param = IterGenMetMomFP(epsi, p, 'Poisson')
lam = param.lam

# ## Initialize projection variables

tau = 10  # investment horizon
dt = 1 / 20  # infinitesimal step for simulations
t_j = arange(0,tau+dt,dt)  # time vector for simulations
j_ = 15  # number of simulations

# ## Simulate Poisson process

k_j = JumpDiffusionMerton(0, 0, lam, 1, 0, t_j, j_)  # generate scenarios
k_j = k_j + k[k_-1]  # centered scenarios

# ## Projection to horizon

# +
# moments
mu_tau = k[k_-1] + lam*tau
sigma_tau = sqrt(lam*tau)

# Poisson pdf at horizon
l_ = int(ceil(mu_tau + 6*sigma_tau))  # number of points
x_pois = arange(0,l_+1)
y_pois = poisson.pmf(x_pois, lam*tau)
x_pois = x_pois + k[k_-1]

# normal approximation
x_norm = linspace(mu_tau - 4*sigma_tau, mu_tau + 4*sigma_tau, l_)
y_norm = norm.pdf(x_norm, mu_tau, sigma_tau)
# -

# ## Generate figure

# +
s_ = 2  # number of plotted observation before projecting time

# axes settings
m = min([npmin(k_j), k[-1]+lam - 2*sigma_tau])
M = max([npmax(k_j), mu_tau + 3.5*sigma_tau])
t = arange(-s_,tau)
max_scale = tau / 4
scale = max_scale / npmax(y_pois)

# preliminary computations
tau_red = arange(0,tau+0.1,0.1)
mu_red = k[-1] + lam*tau_red
sigma_red = sqrt(lam*tau_red)
redline1 = mu_red + 2*sigma_red
redline2 = mu_red - 2*sigma_red

f = figure()
# color settings
lgrey = [0.8, 0.8, 0.8]  # light grey
dgrey = [0.4, 0.4, 0.4]  # dark grey
lblue = [0.27, 0.4, 0.9]  # light blue
plt.axis([t[0], t[-1] + 2*max_scale, m, 1.01*M])
xlabel('time (seconds)')
ylabel('Risk driver')
plt.grid(False)
title('Poisson process')
# simulated paths
for j in range(j_):
    plot(t_j, k_j[j,:], color = lgrey, lw = 2)
# standard deviation lines
p_red_1 = plot(tau_red, redline1, color='r', lw = 2)  # red bars (+2 std dev)
p_red_2 = plot(tau_red, redline2, color='r', lw = 2)  # red bars (-2std dev)
p_mu = plot([0, tau], [k[-1], mu_tau], color='g', lw = 2)  # expectation
# histogram pdf plot
for y in range(len(y_pois)):
    f_hist = plot([tau, tau+y_pois[y]*scale], [x_pois[y], x_pois[y]], color=dgrey, lw=3)
# normal approximation plot
phi_border = plot(tau+y_norm*scale, x_norm, color=lblue, lw=1)
# plot of last s_ observations
for s in range(s_):
    plot([t[s], t[s + 1]], [k[-1- s_ + s - 1], k[- s_ + s - 1]], color=lgrey, lw=2)
    plot(t[s], k[-s_+s-1], color='b',linestyle='none', marker = '.',markersize=15)
plot(t[s_], k[-1], color='b',linestyle='none', marker = '.',markersize=15)
plot([tau, tau], mu_tau + array([-2*sigma_tau, 2*sigma_tau]), color='r', lw = 2)
plt.xticks(arange(-2,11),arange(-2,11))
# leg
#
legend(handles=[p_mu[0], p_red_1[0], f_hist[0], phi_border[0]],labels=['expectation',' + / - 2st.deviation', 'horizon pdf','normal approximation']);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

