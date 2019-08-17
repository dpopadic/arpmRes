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

# # S_ProjectionCompPoisson [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_ProjectionCompPoisson&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerCompPoissExp).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

from numpy import arange, array, ones, zeros, cumsum, round, log, exp, sqrt, unique, where, r_
from numpy import sum as npsum, min as npmin, max as npmax

from scipy.stats import norm
from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, ylabel, \
    xlabel, title, xticks

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict
from FPmeancov import FPmeancov
from HistogramFP import HistogramFP
from EffectiveScenarios import EffectiveScenarios
from IterGenMetMomFP import IterGenMetMomFP
from binningHFseries import binningHFseries
from SimulateCompPoisson import SimulateCompPoisson
from PathMomMatch import PathMomMatch
# -

# ## Upload the database

try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_US_10yr_Future_quotes_and_trades'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_US_10yr_Future_quotes_and_trades'), squeeze_me=True)

# ## Initialize variables

# +
trades = struct_to_dict(db['trades'])

trade_time = trades.time  # time vector of trades
size = trades.siz  # flow of traded contracts' volumes

# set projection variables
tau = 10  # investment horizon
dt = 1 / 20  # infinitesimal step for simulations
t_j = arange(0, tau+dt,dt)  # time vector for simulations
j_ = 3000  # number of simulations
# -

# ## Compute the number of events dn and the traded volume dq at each 1-second interval

# +
t_n = unique(trade_time)
delta_q = zeros((1, len(t_n)))
for k in range(len(t_n)):
    index = trade_time == t_n[k]
    delta_q[0,k] = sum(size[index])  # sum the traded volume relative to the same "match event"

[dn, _, _, dq] = binningHFseries(t_n, '1second', delta_q)  # 1-second spacing
q = cumsum(dq)
# -

# ## Estimate the intensity of Poisson process

# exponential decay FP
lam1 = log(2) / 360
p1 = exp(-lam1 * arange(dn.shape[1],0,-1)).reshape(1,-1)
p1 = p1 / npsum(p1)  # FP-profile: exponential decay 1 years
typ = namedtuple('type','Entropy')
typ.Entropy = 'Exp'
ens1 = EffectiveScenarios(p1, typ)
# generalized method of moments
Parameters = IterGenMetMomFP(dn, p1, 'Poisson')
lam = Parameters.lam

# ## Fit jumps to an exponential distribution

# exponential decay FP
lam2 = log(2) / round(100*lam)
p2 = exp(-lam2 * arange(dq.shape[1],0,-1)).reshape(1,-1)
p2 = p2 / npsum(p2)  # FP-profile: exponential decay 1 years
ens2 = EffectiveScenarios(p2, typ)
# compute FP-mean and variance of an exponential distribution
mu_dq, _ = FPmeancov(dq, p2)
sigma2_dq = mu_dq ** 2

# ## Compute expectation and variance of the compound Poisson process

# +
mu = lam*mu_dq
sigma2 = lam*sigma2_dq
sigma = sqrt(sigma2)

# project to future times
mu_tau = mu*t_j
sigma_tau = sigma*sqrt(t_j)
# -

# ## Simulate the compound Poisson process

# +
method = 'ExpJumps'
c = SimulateCompPoisson(lam, dq, p2, t_j.reshape(1,-1), j_, method)

# path moment-matching via EP
step = int(round(tau / (10*dt)))
p0 = ones((1, j_)) / j_  # initial flat probabilities for the scenarios
c_p = ones((j_, 1))  # constraint on probabilities
c_mu = mu_tau[[0],step::step]  # constraint on expectation
c_sigma2 = sigma_tau[[0],step::step] ** 2  # constraint on variance

p, _ = PathMomMatch(p0, c[:, step::step].T,c_mu.T,c_sigma2.T,c_p.T)

c = c + q[-1]  # centering simulations
# -

# ## Project the pdf to horizon

option = namedtuple('option', 'n_bins')
option.n_bins = 200
y_hor, x_hor = HistogramFP(c[:,[-1]].T, p, option)
# normal approximation
y_norm = norm.pdf(x_hor, q[-1] + mu_tau[0,-1], sigma_tau[0,-1])

# ## Generate figure

# +
s_ = 2  # number of plotted observation before projecting time
j_visual = 15  # number of simulated paths to be printed

# axes settings
c_sample = c[:j_visual,:]
m = min([npmin(c_sample), q[-1]-2*sigma_tau[0,-1]])
M = max([npmax(c_sample), q[-1] + mu_tau[0,-1]+3.5*sigma_tau[0,-1]])  #
t = arange(-s_,tau+1)
max_scale = tau / 4
scale = max_scale / npmax(y_hor)

# preliminary computations
tau_red = arange(0,tau+0.1,0.1)
mu_red = q[-1] + mu*tau_red
sigma_red = sqrt(sigma2*tau_red)
redline1 = mu_red + 2*sigma_red
redline2 = mu_red - 2*sigma_red

f = figure()
# color settings
lgrey = [0.8, 0.8, 0.8]
# light grey
dgrey = [0.55, 0.55, 0.55]
# dark grey
lblue = [0.27, 0.4, 0.9]
# light blue
plt.axis([t[0], t[-1] + max_scale, m, M])
xlabel('time (seconds)')
ylabel('Risk driver')
xticks(r_[t[:s_+ 1], arange(t[-1]+1)])
plt.grid(False)
title('Compound Poisson Process')
# simulated paths
for j in range(j_visual):
    plot(t_j, c[j,:], color = lgrey, lw = 2)
# standard deviation lines
p_red_1 = plot(tau_red, redline1[0], color='r', lw = 2, label='+ / - 2 st.deviation')  # red bars (+2 std dev)
p_red_2 = plot(tau_red, redline2[0], color='r', lw = 2)  # red bars (-2std dev)
p_mu = plot([0, tau], [q[-1], q[-1] + mu_tau[0,-1]], color='g', lw = 2, label='expectation')  # expectation
# histogram pdf plot
for k in range(y_hor.shape[1]):
    f_hist = plot([tau, tau + y_hor[0,k]*scale],[x_hor[k], x_hor[k]], color = dgrey, lw=3, label='horizon pdf')  # normal approximation plot
phi_border = plot(tau + y_norm*scale, x_hor, color=lblue, lw=1, label='Normal approximation')
# plot of last s_ observations
for k in range(s_):
    plot([t[k], t[k+1]], [q[-s_+k-1], q[-s_+k-1]], color=lgrey, lw=2)
    plot(t[k], q[-s_+k-1], color='b',linestyle='none', marker='.',markersize=15)
plot(t[s_], q[-1], color='b',linestyle='none', marker='.',markersize=15)
plot([tau, tau], q[-1]+mu_tau[0,-1]+array([-2*sigma_tau[0,-1], +2*sigma_tau[0,-1]]), color='r', lw=2)
# leg
legend(handles=[f_hist[0],p_red_1[0],p_mu[0], phi_border[0]], labels=['horizon pdf', '+ / - 2 st.deviation','expectation','Normal approximation']);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
