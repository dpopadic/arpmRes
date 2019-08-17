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

# # S_ProjectionVGSub [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_ProjectionVGSub&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-subordinated-brownian-motion).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

from numpy import arange, array, zeros, diff, abs, log, exp, sqrt, tile, r_, atleast_2d, newaxis
from numpy import sum as npsum, min as npmin, max as npmax

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, subplots, ylabel, \
    xlabel, title, xticks

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import struct_to_dict, datenum, save_plot
from intersect_matlab import intersect
from EffectiveScenarios import EffectiveScenarios
from ConditionalFP import ConditionalFP
from MMFP import MMFP
from VG import VG
from ShiftedVGMoments import ShiftedVGMoments
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

# ## Merge data

# +
# invariants (daily P&L)
pnl = OptionStrategy.cumPL
epsi = diff(pnl)
dates_x = array([datenum(i) for i in OptionStrategy.Dates])
dates_x = dates_x[1:]

# conditioning variable (VIX)
z = VIX.value
dates_z = VIX.Date

    # merging datasets
[dates, i_epsi, i_z] = intersect(dates_x, dates_z)

pnl = pnl[i_epsi + 1]
epsi = epsi[i_epsi]
z = z[i_z]
t_ = len(epsi)
# -

# ## Compute the Flexible Probabilities conditioned via Entropy Pooling

# +
# prior
lam = log(2) / 1800  # half life 5y
prior = exp(-lam*abs(arange(t_, 1 + -1, -1))).reshape(1,-1)
prior = prior / npsum(prior)

# conditioner
VIX = namedtuple('VIX', 'Series TargetValue Leeway')
VIX.Series = z.reshape(1,-1)
VIX.TargetValue = atleast_2d(z[-1])
VIX.Leeway = 0.35

# flexible probabilities conditioned via EP
p = ConditionalFP(VIX, prior)

# effective number of scenarios
typ = namedtuple('type','Entropy')
typ.Entropy = 'Exp'
ens = EffectiveScenarios(p, typ)
# -

# ## Estimation of shifted-VG model

# +
# initial guess on parameters
shift0 = 0
theta0 = 0
sigma0 = 0.01
nu0 = 1
par0 = [shift0, theta0, sigma0, nu0]

# calibration
HFP = namedtuple('HFP', ['FlexProbs','Scenarios'])
HFP.FlexProbs = p
HFP.Scenarios = epsi
par = MMFP(HFP, 'SVG', par0)

shift = par.c
theta = par.theta
sigma = par.sigma
nu = par.nu

# #changing parameterization from {theta,sigma, nu} to {c,m,g}
# [c, m, g] = ParamChangeVG(theta,sigma,nu)
# -

# ## Initialize projection variables
tau = 15  # investment horizon
dt = 1 / 75  # infinitesimal step for simulations
t_j = arange(0,tau+dt,dt)  # time vector for simulations
j_ = 2  # number of simulations

# +
# ## Simulate VG paths

[X, T] = VG(theta, sigma, nu, t_j, j_)  # VG paths
X = X + tile(shift*t_j[newaxis,...], (j_, 1))  # shifted-VG path
X = pnl[t_-1] + X  # centered path
dT = r_['-1',zeros((j_, 1)), diff(T, 1, 1)]
# -

# ## Projection to horizon

# moments
mu_tau, sigma2_tau, _, _ = ShiftedVGMoments(0, theta, sigma, nu, tau)
expectation = pnl[t_-1] + shift*tau + mu_tau  # shift and center mean
sigma_tau = sqrt(sigma2_tau)

# ## Generate the figure
s_ = 2

# +
f, ax = subplots(3,1)

# figure settings
dgrey = [0.5, 0.5, 0.5]
color = {}
color [0]= 'b'
color [1]= [.9, .35, 0]
color [2]= 'm'
color [3]= 'g'
color [4]= 'c'
color [5]= 'y'
t = r_[arange(-s_,1),t_j[1:]]

plt.sca(ax[0])
m = min([npmin(X)*0.91, npmin(pnl[t_ - s_:])*0.91, pnl[-1] - 3*sigma_tau / 2])
M = max([npmax(X)*1.1, npmax(pnl[t_ - s_:])*1.1, expectation + 1.2*sigma_tau])
plt.axis([-s_, tau, m, M])
xlabel('time (days)')
ylabel('Risk driver')
xticks(arange(-s_,tau+1))
plt.grid(False)
title('Variance Gamma process (subordinated Brownian motion)')
for j in range(j_):
    plot(t_j, X[j,:], color= color[j], lw=2)

for s in range(s_):
    plot([s-s_, s-s_+1], [pnl[t_+s-s_-1], pnl[t_+s-s_]], color=dgrey, lw=2)
    plot(s-s_, pnl[t_+s-s_-1], color=dgrey, linestyle='none', marker='.',markersize=15) # observation (dots)

plot(0, pnl[t_-1], color=dgrey, linestyle='none', marker='.',markersize=15)

plt.sca(ax[1])
M_v = npmax(dT)*1.1
m_v = -M_v*0.08
plt.axis([-s_, tau, m_v, M_v])
xlabel('time (days)')
ylabel('Stoch. time increment')
xticks(arange(-s_,tau+1))
plt.grid(False)
title('Gamma process')
for j in range(j_):
    plot(t_j, dT[j,:], color= color[j], lw=2)

plot([-s_, 0], [0,0], color=dgrey, lw=2)

plt.sca(ax[2])
M_T = npmax(T[:,-1])*1.1
m_T = -M_T*0.08
plt.axis([-s_, tau, m_T, M_T])
xlabel('time (days)')
ylabel('Stoch. time')
xticks(arange(-s_,tau+1))
plt.grid(False)
title('Integrated Gamma process')
for j in range(j_):
    plot(t_j, T[j,:], color= color[j], lw=2)

plot([-s_, 0], [0,0], color=dgrey, lw=2)
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1]);
