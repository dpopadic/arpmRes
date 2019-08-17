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

# # S_ProjectionVG [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_ProjectionVG&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExVGProj).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

from numpy import arange, array, diff, linspace, abs, log, exp, sqrt, tile, atleast_2d, newaxis
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
from VGpdf import VGpdf
from ParamChangeVG import ParamChangeVG
from ShiftedVGMoments import ShiftedVGMoments
from VG import VG
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
shift0 = 0.01
theta0 = -0.01
sigma0 = 0.0001
nu0 = 1
par0 = [shift0, theta0, sigma0, nu0]

# calibration
HFP = namedtuple('HFP', 'FlexProbs Scenarios')
HFP.FlexProbs = p
HFP.Scenarios = epsi
par = MMFP(HFP, 'SVG', par0)

shift = par.c
theta = par.theta
sigma = par.sigma
nu = par.nu

# changing parameterization from {theta,sigma, nu} to {c,m,g}
[c, m, g] = ParamChangeVG(theta, sigma, nu)
# -

# ## Initialize projection variables

tau = 10  # investment horizon
dt = 1 / 20  # infinitesimal step for simulations
t_j = arange(0,tau+dt,dt)  # time vector for simulations
j_ = 15  # number of simulations

# ## Simulate VG paths

x_j = VG(theta, sigma, nu, t_j, j_)[0]  # VG paths
x_j = x_j + tile(shift*t_j[newaxis,...],(j_, 1))  # shifted-VG path
x_j = pnl[t_-1] + x_j  # centered path

# ## Projection to horizon

# +
# moments
mu_tau, sigma2_tau, _, _ = ShiftedVGMoments(0, theta, sigma, nu, tau)
expectation = pnl[t_-1] + shift*tau + mu_tau  # shift and center mean
sigma_tau = sqrt(sigma2_tau)

# analytical pdf
l_ = 2000
par1 = namedtuple('par', 'c m g')
par1.c = c
par1.m = m
par1.g = g
x_hor = linspace(mu_tau - 4*sigma_tau, mu_tau+4*sigma_tau, l_)
y_hor = VGpdf(x_hor, par1, tau)
y_phi = norm.pdf(x_hor, mu_tau, sigma_tau)  # normal approximation

x_shift = x_hor + pnl[t_-1] + shift*tau
# -

# ## Generate figure

# +
s_ = 2  # number of plotted observation before projecting time

# axes settings
m = min([npmin(pnl[t_ - 2:t_]), pnl[t_-1]-4*sigma_tau])
M = max([npmax(pnl[t_ - 2:t_]), pnl[t_-1] + mu_tau + 4.5*sigma_tau])
t = arange(-s_, tau+1)
max_scale = tau / 4
scale = max_scale / npmax(y_hor)

# preliminary computations
tau_red = arange(0,tau+0.1,0.1)
mu_red = pnl[-1] + ((mu_tau + shift*tau) / tau)*tau_red
sigma_red = (sigma_tau / sqrt(tau))*sqrt(tau_red)
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
title('Variance Gamma process')
# simulated paths
for j in range(j_):
    plot(t_j, x_j[j,:], color = lgrey, lw = 2)
# standard deviation lines
p_red_1 = plot(tau_red, redline1, color='r', lw = 2)  # red bars (+2 std dev)
p_red_2 = plot(tau_red, redline2, color='r', lw = 2)  # red bars (-2std dev)
p_mu = plot([0, tau], [mu_red[0], mu_red[-1]], color='g', lw = 2)  # expectation
# histogram pdf plot
for k in range(len(y_hor)):
    plot([tau, tau+y_hor[k]*scale], [x_shift[k], x_shift[k]], color=lgrey, lw=2)
f_border = plot(tau+y_hor*scale, x_shift, color=dgrey, lw=1)
# normal approximation plot
phi_border = plot(tau+y_phi*scale, x_shift, color=lblue, lw=1)
# plot of last s_ observations
for k in range(s_):
    plot([t[k], t[k + 1]], [pnl[-s_+k-1], pnl[- s_ + k]], color=lgrey, lw=2)
    plot(t[k], pnl[-s_+k-1], color='b',linestyle='none', marker = '.',markersize=15)
plot(t[s_], pnl[-1], color='b',linestyle='none', marker = '.',markersize=15)
plot([tau, tau], expectation + array([-2*sigma_tau, 2*sigma_tau]), color='r', lw = 2)
legend(handles=[f_border[0], phi_border[0], p_mu[0], p_red_1[0]],labels=['horizon pdf','normal approximation','expectation',' + / - 2st.deviation']);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

