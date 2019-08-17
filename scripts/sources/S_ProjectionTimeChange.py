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

# # S_ProjectionTimeChange [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_ProjectionTimeChange&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-time-changed-browiona-motion).

# ## Prepare the environment

# +
import os
import os.path as path
import sys
from collections import namedtuple

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import arange, array, zeros, cumsum, diff, abs, log, exp, sqrt, tile, r_
from numpy import sum as npsum, min as npmin, max as npmax
from numpy.random import multivariate_normal as mvnrnd

from scipy.stats import t
from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, ylabel, \
    xlabel, title

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import struct_to_dict, save_plot
from FPmeancov import FPmeancov
from StochTime import StochTime
from FitCIR_FP import FitCIR_FP
# -

# ## Upload databases

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_Stocks'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_Stocks'), squeeze_me=True)

SPX = struct_to_dict(db['SPX'])
# -

# ## Compute the time series of risk driver

date = SPX.Date
x = log(SPX.Price_close).reshape(1,-1)
dx = diff(x)

# ## Estimate realized variance

# +
s_ = 252  # forward/backward parameter
lambda1 = log(2) / 5  # half-life one week
p1 = exp(-lambda1*abs(arange(-s_,s_+1))).reshape(1,-1)
p1 = p1 / npsum(p1)

t_var = dx.shape[1] - 2*s_
y = zeros((1, t_var))
for s in range(t_var):
    dx_temp = dx[0,s:s + 2*s_+1]
    y[0,s] = p1@(dx_temp.T ** 2)

# daily variance
dx = dx[:,s_ :s_ + t_var]
x = x[:,s_ :s_ + t_var + 1]
# -

# ## Calibrate the CIR process

# +
t_obs = 252*4  # 4 years
lambda2 = log(2) / (21*9)  # half-life 9 months
p2 = exp(-lambda2*arange(t_obs, 1 + -1, -1)).reshape(1,-1)
p2 = p2 / npsum(p2)

delta_t = 1  # fix the unit time-step to 1 day

par_CIR = FitCIR_FP(y[0,- t_obs:], delta_t, None, p2)

kappa = par_CIR[0]
y_ = par_CIR[1]
eta = par_CIR[2]
# -

# ## Estimate mu (drift parameter of X) and rho (correlation between Brownian motions)

# +
dy = diff(y)
xy = r_[dx[:,- t_obs:], dy[:,- t_obs:]]
[mu_xy, sigma2_xy] = FPmeancov(xy, p2)  # daily mean vector and covariance matrix

mu = mu_xy[0]  # daily mean
rho = sigma2_xy[0, 1] / sqrt(sigma2_xy[0, 0]*sigma2_xy[1, 1])  # correlation parameter
# -

# ## Initialize projection variables

dt = 1  # two days
tau = 5*252  # ten years
t_j = arange(0,tau+dt,dt)
t_sim = len(t_j) - 1
j_ = 2

# ## Simulate paths

# +
# initialize variables
Y = zeros((j_, t_sim + 1))
T = zeros((j_, t_sim + 1))
dT = zeros((j_, t_sim))
dX = zeros((j_, t_sim))
Y[:, [0]] = tile(y[[0],-1], (j_, 1))

# initialize inputs for stoch. time function
inp = namedtuple('inp','kappa s2_ eta S2_t z')
inp.kappa = kappa
inp.s2_ = y_
inp.eta = eta

# Euler scheme
for t in range(t_sim):
    W = mvnrnd([0,0], array([[1, rho],[rho, 1]]), j_)
    inp.S2_t = Y[:,t]
    inp.z = W[:, 0]
    dT[:,t] = StochTime(dt, 'Heston', inp)  # stochastic time
    Y[:, t + 1] = dT[:,t] / dt  # variance process
    T[:, t + 1] = T[:,t]+dT[:,t]  # time-change process
    dX[:,t] = mu*dt + sqrt(dT[:,t])*W[:, 1]

X = x[:,-1] + r_['-1',zeros((j_, 1)), cumsum(dX, 1)]
# -

# ## Generate the figure

# +
s_ = 1

# figure settings
dgrey = [0.5, 0.5, 0.5]
color = {}
color [0]= 'b'
color [1]= [.9, .35, 0]
color [2]= 'm'
color [3]= 'g'
color [4]= 'c'
color [5]= 'y'
tau_plot = tau / 252
t = arange(-s_,tau_plot+1/25, 1/25)

f, ax = plt.subplots(3, 1)
plt.sca(ax[0])
m_x = min([npmin(X), npmin(x[:,-252*s_:])])*.9
M_x = max([npmax(X), npmax(x[:,-252*s_:])])*1.1
plt.axis([-s_, tau_plot, m_x, M_x])
ylabel('Risk driver')
plt.grid(False)
title('Heston process (Stochastic time-changed Brownian motion)')
for j in range(j_):
    plot(t_j / 252, X[j,:], color= color[j], lw=2)

for s in range(s_*252 + 1):
    plot(-s_+(s-1) / 252, x[:,-252*s_+s-1], color=dgrey, lw=2)

plt.sca(ax[1])
m_y = min([npmin(Y), npmin(y[:,- 252*s_:])])*.9
M_y = max([npmax(Y), npmax(y[:,- 252*s_:])])*1.1
range_y = M_y - m_y
m_y = m_y - range_y*.15
M_y = M_y + range_y*.15
plt.axis([-s_, tau_plot, m_y, M_y])
ylabel('Stoch. variance')
plt.grid(False)
title('Square-root (CIR) process')
for j in range(j_):
    plot(t_j / 252, Y[j,:], color= color[j], lw=2)

for s in range(s_*252 + 1):
    plot(-s_+(s-1) / 252, y[:,-252*s_+s-1], color=dgrey, lw=2)

plt.sca(ax[2])
m_t = -npmax(T)*.1
M_t = npmax(T)*1.1
plt.axis([-s_, tau_plot, m_t, M_t])
xlabel('time (years)')
ylabel('Stoch. time')
plt.grid(False)
title('Integrated square-root process')
for j in range(j_):
    plot(t_j / 252, T[j,:], color= color[j], lw=2)

for s in range(s_*252 + 1):
    plot(-s_+(s-1) / 252, 0, color=dgrey, lw=2)
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

