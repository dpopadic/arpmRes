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

# # S_ProjectionLogNorm [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_ProjectionLogNorm&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-lognormal-projection).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import arange, array, ones, diff, abs, log, exp, sqrt, r_, real
from numpy import sum as npsum, min as npmin, max as npmax

from scipy.stats import norm
from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, ylabel, \
    xlabel, title

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import struct_to_dict, datenum, save_plot
from intersect_matlab import intersect
from EffectiveScenarios import EffectiveScenarios
from ConditionalFP import ConditionalFP
from MMFP import MMFP
from ShiftedLNMoments import ShiftedLNMoments
from ProjDFFT import ProjDFFT
# -

# ## Upload databases db_OptionStrategy and db_VIX

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

# ## Compute the invariants and the time series of the conditioning variable (VIX index)

#invariants
x = OptionStrategy.cumPL
dx = diff(x)
dates_x = array([datenum(i) for i in OptionStrategy.Dates])
dates_x = dates_x[1:]
# conditioning variable (VIX)
z = VIX.value
dates_z = VIX.Date

# ## Intersect the two database to obtain data corresponding to the same dates

[dates, i_dx, i_z] = intersect(dates_x, dates_z)
x = x[i_dx + 1]
dx = dx[i_dx]
z = z[i_z]
t_ = len(dx)

# ## Compute the Flexible Probabilities by mixing an exponential decay prior with the current information on the VIX

# +
#prior
lam = log(2) / 1080  # half life 3y
prior = exp(-lam*abs(arange(t_, 1 + -1, -1))).reshape(1,-1)
prior = prior / npsum(prior)

# conditioner
VIX = namedtuple('VIX', 'Series TargetValue Leeway')
VIX.Series = z.reshape(1,-1)
VIX.TargetValue = np.atleast_2d(z[-1])
VIX.Leeway = 0.35

# flexible probabilities conditioned via EP
p = ConditionalFP(VIX, prior)

p[p == 0] = 10 ** -20  # avoid log[0-1] in ens computation
p = p /npsum(p)

# effective number of scenarios
typ = namedtuple('type','Entropy')
typ.Entropy = 'Exp'
ens = EffectiveScenarios(p, typ)
# -

# ## Fit the reflected shifted lognormal distribution by using function MMFP

# +
# compute skewness
m1 = p@dx.T
m2 = p@((dx - m1) ** 2).T
m3 = p@((dx - m1) ** 3).T
skewness = m3 / (m2) ** 1.5

HFP = namedtuple('HFP', ['FlexProbs','Scenarios'])
HFP.FlexProbs = p
HFP.Scenarios = dx
parameters = MMFP(HFP, 'SLN')
# -

# ## Compute the expectation and standard deviation of the fitted distribution by using function ShiftedLNMoments

parameters.skew = skewness
mu, sigma,_ = ShiftedLNMoments(parameters)
mu = real(mu)
sigma = real(sigma)

# ## Project the expectation and standard deviation

# +
tau = 20  # investment horizon
k_ = 2 ** 11

mu_tau = x[t_-1] + mu*tau
sigma_tau = sigma*sqrt(tau)
# -

# ## Project the estimated pdf to the horizon via the FFT algorithm

x_hat_hor, f_hat_hor,*_ = ProjDFFT(None, None, tau, k_, 'shiftedLN', parameters)
x_hor = mu*tau*ones((1, len(x_hat_hor))) + sigma*x_hat_hor
f_hor = f_hat_hor / sigma
f_hor = np.real(f_hor)

# ## Compute the normal approximation of the projected pdf

phi_hor = norm.pdf(x_hor, mu*tau, sigma*sqrt(tau))
# center around x(t_end)
x_hor = x_hor + x[t_-1]

# ## Create a figure

# +
s_ = 2  # number of plotted observation before projecting time

# axes settings
m = min([npmin(x[t_ - s_:t_]), mu_tau - 4.5*sigma_tau])
M = max([npmax(x[t_ - s_:t_]), mu_tau + 5*sigma_tau])
t = arange(-s_,tau)
max_scale = tau / 4

# preliminary computations
tau_red = arange(0,tau+0.1,0.1)
mu_red = x[t_-1] + mu*tau_red
sigma_red = sigma*sqrt(tau_red)
redline1 = mu_red + 2*sigma_red
redline2 = mu_red - 2*sigma_red

f = figure()

# color settings
lgrey = [0.8, 0.8, 0.8]  # light grey
dgrey = [0.2, 0.2, 0.2]  # dark grey
lblue = [0.27, 0.4, 0.9]  # light blue

plt.axis([t[0], t[-1] + 1.3*max_scale, np.real(m), np.real(M)])
xlabel('time (days)')
ylabel('Risk driver')
plt.xticks(arange(-2, 21))
#  'XTick', [t(range(s_) + 1) range(t)[-1]], 'XTickLabel', num2str([t(range(s_) + 1) range(t)[-1]]
# ',.T#1.0f')
plt.grid(False)
title('Negative shifted log-normal projection')

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

# arrows
plot([tau, tau], mu_tau + array([-2*sigma_tau, 2*sigma_tau]), color='r', lw = 2)

# leg
legend(handles=[f_border[0], phi_border[0], p_mu[0], p_red_1[0]],
       labels=['horizon pdf','normal approximation','expectation',' + / - 2 st.deviation']);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
