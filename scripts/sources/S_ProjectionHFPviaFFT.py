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

# # S_ProjectionHFPviaFFT [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_ProjectionHFPviaFFT&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerHFPProj).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import arange, reshape, ones, cumsum, diff, abs, log, exp, sqrt, array, r_
from numpy import sum as npsum, min as npmin, max as npmax

from scipy.stats import norm
from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, ylabel, \
    xlabel, title

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import struct_to_dict, datenum, save_plot
from FPmeancov import FPmeancov
from intersect_matlab import intersect
from EffectiveScenarios import EffectiveScenarios
from ConditionalFP import ConditionalFP
from ProjDFFT import ProjDFFT
from SampleScenProbDistribution import SampleScenProbDistribution

# parameters
j_ = 10 ** 4  # Number of scenarios
deltat = 20  # investment horizon expressed in days
tau_HL = 1260  # Half life probability expressed in days
alpha = 0.35  # probability range
k_ = 2 ** 12  # coarseness level for projection
# -

# ## Upload databases

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_OptionStrategy'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_OptionStrategy'), squeeze_me=True)

OptionStrategy = struct_to_dict(db['OptionStrategy'])

try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_VIX'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_VIX'), squeeze_me=True)

VIX = struct_to_dict(db['VIX'])
# -

# ## Compute the realized time series of daily P&L

# +
# invariants (daily P&L)
pi = OptionStrategy.cumPL  # cumulative P&L
epsi = diff(pi)  # daily P&L
dates_x = array([datenum(i) for i in OptionStrategy.Dates])
dates_x = dates_x[1:]

# conditioning variable (VIX)
z = VIX.value
dates_z = VIX.Date
# -

# ## Intersect the time series of daily P&L and VIX

# +
dates, i_epsi, i_z = intersect(dates_x, dates_z)

pi = pi[i_epsi + 1]
epsi = epsi[i_epsi]
z = z[i_z]
t_ = len(epsi)
# -

# ## Perform the Minimum Relative Entropy Pooling conditioning

# +
# prior
lam = log(2) / tau_HL  # half life 5y
prior = exp(-lam*abs(arange(t_, 1 + -1, -1))).reshape(1,-1)
prior = prior / npsum(prior)

# conditioner
VIX = namedtuple('VIX', 'Series TargetValue Leeway')
VIX.Series = z.reshape(1,-1)
VIX.TargetValue = np.atleast_2d(z[-1])
VIX.Leeway = alpha

# flexible probabilities conditioned via EP
p = ConditionalFP(VIX, prior)
p[p == 0] = 10 ** -20  # avoid log[0-1] in ens computation
p = p /npsum(p)

# effective number of scenarios
typ = namedtuple('type','Entropy')
typ.Entropy = 'Exp'
ens = EffectiveScenarios(p, typ)
# -

# ## Compute the HFP-estimators of location and dispersion

mu, sigma2 = FPmeancov(epsi.reshape(1,-1), p)
sigma = sqrt(sigma2)

# ## Project the HFP-pdf to the horizon

# +
[mu_hat, sigma2_hat] = FPmeancov(epsi.reshape(1,-1), p)
sigma_hat = sqrt(sigma2_hat)
epsi_hat = (epsi - mu_hat) / sigma_hat

xi, f_hat_deltat,*_ = ProjDFFT(epsi_hat, p, deltat, k_)
epsi_deltat = mu_hat*deltat + sigma_hat*xi
f_deltat = f_hat_deltat / sigma_hat
f_deltat = np.real(f_deltat)
# -

# ## Compute a normal approximation of the projected HFP-pdf

# +
phi_deltat = norm.pdf(epsi_deltat, mu*deltat, sigma*sqrt(deltat))

# center around last realization of pi
epsi_deltat = epsi_deltat + pi[-1]
# -

# ## (optional) compute low-order central moments at the horizon

# +
# compute one-step FP-moments
mu_1 = mu.squeeze()  # expected value
sigma_1 = sigma.squeeze()  # standard deviation
varsigma_1 = p@(epsi.reshape(-1,1) - mu_1) ** 3 / (sigma_1 ** 3)  # skewness
kappa_1 = p@(epsi.reshape(-1,1) - mu_1) ** 4 / (sigma ** 4) - 3  # excess kurtosis

# project FP-moments to horizon tau
mu_tau = pi[-1] + mu_1*deltat  # projected (shifted) expexted value
sigma_tau = sigma_1*sqrt(deltat)  # projected standard deviation
varsigma_tau = varsigma_1 / sqrt(deltat)  # projected skewness
kappa_tau = kappa_1 / deltat  # projected excess kurtosis
# -

# ## Simulate scenarios of projected path risk drivers

# +
# Generate scenarios for the invariants via historical bootstrapping
epsi_hor = SampleScenProbDistribution(epsi.reshape(1,-1), p, j_*deltat)
epsi_hor = reshape(epsi_hor, (j_, deltat),'F')

# Feed the simulated scenarios in the recursive incremental-step routine((random walk assumption))
pi_deltat = pi[-1] + cumsum(epsi_hor, 1)
# -

# ## Create a figure

# +
s_ = 2  # number of plotted observation before projecting time

# axes settings
m = min([npmin(pi[::-1]), pi[-1]-4*sigma_tau])
M = max([npmax(pi[::-1]), mu_tau + 4.5*sigma_tau])
t = arange(-s_,deltat+1)
max_scale = deltat / 4

# preliminary computations
tau_red = arange(0,deltat+0.1,0.1)
mu_red = pi[-1] + mu_1*tau_red
sigma_red = sigma_1*sqrt(tau_red)
redline1 = mu_red + 2*sigma_red
redline2 = mu_red - 2*sigma_red

from matplotlib.pyplot import xticks

f = figure()
# color settings
lgrey = [0.8, 0.8, 0.8]  # light grey
dgrey = [0.2, 0.2, 0.2]  # dark grey
lblue = [0.27, 0.4, 0.9]  # light blue
plt.axis([t[0], t[-1] + max_scale, m, M])
xlabel('time (days)')
ylabel('Risk driver')
plt.xticks(arange(-2,21))
plt.grid(False)
title('Historical process with Flexible Probabilities')
# standard deviation lines
p_red_1 = plot(tau_red, redline1, color='r', lw=2)  # red bars (+2 std dev)
p_red_2 = plot(tau_red, redline2, color='r', lw=2)  # red bars (-2std dev)
p_mu = plot([0, deltat], [pi[-1], mu_tau], color='g', lw = 2)  # expectation
# histogram pdf plot
for k in range(f_deltat.shape[1]):
    plot([deltat, deltat + f_deltat[0,k]], [epsi_deltat[0,k], epsi_deltat[0,k]], color=lgrey, lw=2)
f_border = plot(deltat + f_deltat.T, epsi_deltat.T, color=dgrey, lw=1)
# normal approximation plot
phi_border = plot(deltat + phi_deltat.T, epsi_deltat.T, color=lblue, lw=1)
# plot of last s_ observations
for k in range(s_):
    plot([t[k], t[k+1]], [pi[-s_+k-1], pi[-s_+k]], color=lgrey, lw=2)
    plot(t[k], pi[-s_+k-1], color='b',linestyle='none', marker='.',markersize=15)
plot(t[s_], pi[-1], color='b',linestyle='none', marker = '.',markersize=15)
# paths
plot(arange(deltat+1), r_['-1',pi[-1]*ones((20, 1)), pi_deltat[:20,:]].T, color= lgrey, lw=1,zorder=0)
# leg
legend(handles=[f_border[0], phi_border[0], p_mu[0], p_red_1[0]],
       labels=['horizon pdf','normal approximation','expectation',' + / - 2 st.deviation']);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])


