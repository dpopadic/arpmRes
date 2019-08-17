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

# # S_ProjectionUniform [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_ProjectionUniform&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExUniformProjection).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

from numpy import arange, array, ones, sqrt, real
from numpy import min as npmin, max as npmax

from scipy.stats import norm
from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, ylabel, \
    xlabel, title

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict
from ProjDFFT import ProjDFFT
# -

# ## Upload databases db_Uniform

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_Uniform'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_Uniform'), squeeze_me=True)

UniformStrategy = struct_to_dict(db['UniformStrategy'])
# -

# ## Select the time series of cumulative P&L and set an horizon tau = 20

# +
x = UniformStrategy.cumPL

t_ = 4  #
tau = 20  # investment horizon expressed in days
k_ = 2**11  # coarseness level for projection
# -

# ## Set the parameters of the one-step uniform distribution

mu = 1/2  # mean
sigma = 1 / sqrt(12)  # standard deviation
par = namedtuple('par', 'mu sigma')
par.mu = mu
par.sigma = sigma

# ## Compute the projected moments to the horizon

#moments to horizon
mu_tau = mu*tau
sigma_tau = sigma*sqrt(tau)

# ## Use function ProjFFT to compute the projected pdf to the horizon

epsi_hat_tau, f_hat_tau,_ = ProjDFFT(None, None, tau, k_, 'Uniform', par)
epsi_tau = mu_tau*ones((1, len(epsi_hat_tau))) + sigma*epsi_hat_tau
f_tau = f_hat_tau / sigma
f_tau = real(f_tau)

# ## Compute the normal approximation of the pdf

# +
phi_tau = norm.pdf(epsi_tau, mu*tau, sigma*sqrt(tau))

# center around x[t_end-1]
epsi_tau = epsi_tau + x[t_-1]
# -

# ## Create a figure

# +
s_ = 2  # number of plotted observation before projecting time

# axes settings
m = min([npmin(x[t_ - 2:t_]), x[t_]-4*sigma_tau])
M = max([npmax(x[t_ - 2:t_]), x[t_] + mu_tau + 4.5*sigma_tau])
t = arange(-s_,tau+1)
max_scale = tau / 4
scale = max_scale / npmax(f_tau)

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
plt.axis([t[0], t[-1] + 1.3*max_scale, m, M])
xlabel('time (days)')
ylabel('Risk driver')
plt.grid(False)
title('Uniform distribution projection')
plt.xticks(arange(-2, 21))
# standard deviation lines
p_red_1 = plot(tau_red, redline1, color='r', lw = 2)  # red bars (+2 std dev)
p_red_2 = plot(tau_red, redline2, color='r', lw = 2)  # red bars (-2std dev)
p_mu = plot([0, tau], [x[t_-1], x[t_-1] + mu_tau], color='g', lw = 2)  # expectation
# histogram pdf plot
for k in range(len(f_tau)):
    plot([tau, tau+f_tau[k]*scale], [epsi_tau[0,k], epsi_tau[0,k]], color=lgrey, lw=2)
f_border = plot(tau+f_tau*scale, epsi_tau[0], color=dgrey, lw=1)
# normal approximation plot
phi_border = plot(tau+phi_tau[0]*scale, epsi_tau[0], color=lblue, lw=1)
# plot of last s_ observations
for k in range(s_):
    plot([t[k], t[k + 1]], [x[t_ - s_ + k - 1], x[t_ - s_ + k]], color=lgrey, lw=2)
    plot(t[k], x[t_ - s_ + k - 1], color='b',linestyle='none', marker = '.',markersize=15)
plot(t[s_], x[t_-1], color='b',linestyle='none', marker = '.',markersize=15)
plot([tau, tau], x[t_-1] + mu_tau + array([-2*sigma_tau, 2*sigma_tau]), color='r', lw = 2)
legend(handles=[f_border[0], phi_border[0], p_mu[0], p_red_1[0]],labels=['horizon pdf','normal approximation','expectation',' + / - 2st.deviation']);
plt.show()
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
