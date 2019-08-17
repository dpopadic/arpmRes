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

# # S_ProjectionHeston [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_ProjectionHeston&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-estimation-cirmfp).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

from numpy import arange, array, ones, zeros, cumsum, diff, linspace, abs, log, exp, sqrt, tile, r_
from numpy import sum as npsum, min as npmin, max as npmax
from numpy.random import randn

from scipy.stats import gamma
from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, legend, subplots, ylabel, \
    xlabel, title

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict
from FPmeancov import FPmeancov
from HistogramFP import HistogramFP
from PathMomMatch import PathMomMatch
from FitCIR_FP import FitCIR_FP
from HestonChFun import HestonChFun
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
x = x[:,s_ :s_ + t_var+1]
# -

# ## Calibrate the CIR process

# +
t_obs = 252*4  # 4 years
lambda2 = log(2) / (21*9)  # half-life 9 months
p2 = exp(-lambda2*arange(t_obs, 1 + -1, -1)).reshape(1,-1)
p2 = p2 / npsum(p2)

delta_t = 1  # fix the unit time-step to 1 day

par_CIR = FitCIR_FP(y[0,-t_obs:], delta_t, None, p2)

kappa = par_CIR[0]
y_ = par_CIR[1]
eta = par_CIR[2]
# -

# ## Estimate mu (drift parameter of X) and rho (correlation between Brownian motions)

# +
dy = diff(y)
xy = r_[dx[:,-t_obs:], dy[:,- t_obs:]]
[mu_xy, sigma2_xy] = FPmeancov(xy, p2)  # daily mean vector and covariance matrix

mu = mu_xy[0]  # daily mean
rho = sigma2_xy[0, 1] / sqrt(sigma2_xy[0, 0]*sigma2_xy[1, 1])  # correlation parameter
# -

# ## Initialize projection variables

dt = 2  # two days
tau = 2*252  # two years
t_j = arange(0,tau+dt,dt)
t_sim = len(t_j) - 1
j_ = 3000

# ## Simulate paths

# +
dW_1 = tile(sqrt(diff(t_j)), (j_, 1))*randn(j_, t_sim)
dW_uncorr = tile(sqrt(diff(t_j)), (j_, 1))*randn(j_, t_sim)
dW_2 = rho*dW_1 + sqrt(1 - rho ** 2)*dW_uncorr

# initialize arrays
dY = zeros((j_, t_sim))
Y = zeros((j_, t_sim + 1))
dX = zeros((j_, t_sim))

# Euler scheme
Y[:, [0]] = y[:,-1]*ones((j_, 1))  # initialize variance
for t in range(t_sim):
    dY[:,t] = -kappa*(Y[:,t]-y_)*dt + eta*sqrt(Y[:,t])*dW_2[:,t]
    Y[:, t + 1] = abs((Y[:,t]+dY[:,t]))
    dX[:,t] = mu*dt + sqrt(Y[:,t])*dW_1[:,t]

X = x[:,-1] + r_['-1',zeros((j_, 1)), cumsum(dX, 1)]
# -

# ## Compute analytical first and second moments via characteristic function
# ##syms z x1 x2 x3 x4 x5 x6 x7 x8
# ##f(z, x1, x2, x3, x4, x5, x6, x7, x8) = HestonChFun((z/1i, x1, x2, x3, x4, x5, x6, x7, x8))
# ##mu1(z, x1, x2, x3, x4, x5, x6, x7, x8) = diff(f, z, 1)
# ##mu2(z, x1, x2, x3, x4, x5, x6, x7, x8) = diff(f, z, 2)
# ##mu_x = zeros((1,t_sim+1))
# ##sigma2_x = zeros((1,t_sim+1))
# ##for t in range(t_sim+1
# ##    mu_x[t] = subs((mu1(0,mu,kappa,y_,eta,rho,x[:,-1],y[:,-1],t_j[t])))
# ##    sigma2_x[t] = subs((mu2(0,mu,kappa,y_,eta,rho,x[:,-1],y[:,-1],t_j[t])) - mu_x[t])**2

# +
delta = 0.001
mu_x = zeros(t_sim+1)
sigma2_x = zeros(t_sim+1)
for t in range(t_sim + 1):
    mu_x[t] = (HestonChFun(delta / 1j, mu, kappa, y_, eta, rho, x[:,-1], y[:,-1], t_j[t])
               -HestonChFun(-delta / 1j, mu, kappa, y_, eta, rho, x[:,-1], y[:,-1], t_j[t]))[0] / (2*delta)
    sigma2_x[t] = - mu_x[t]**2+(HestonChFun(delta / 1j, mu, kappa, y_, eta, rho, x[:,-1], y[:,-1], t_j[t])
                                -2*HestonChFun(0, mu, kappa, y_, eta, rho, x[:,-1], y[:,-1], t_j[t])
                                +HestonChFun(-delta / 1j, mu, kappa, y_, eta, rho, x[:,-1], y[:,-1], t_j[t]))[0] / (delta ** 2)

    sigma2_x[sigma2_x < 0] = 0

# exact moments of CIR process
mu_y = y[:,-1]*exp(-kappa*t_j) + y_*(1-exp(-kappa*t_j))
sigma2_y = y[:,-1]*(eta ** 2 / kappa)*( exp(-kappa*t_j) - exp(-2*kappa*t_j) ) + ((y_*eta ** 2) / (2*kappa))*(1-exp(-kappa*t_j)) ** 2
# -

# ## Path Moment-Matching

# +
step = 21
q = ones((1, j_)) / j_  # initial flat probabilities
p_constr = ones((j_, 1))  # constraint on probabilities
mu_constr = mu_x[step::step].reshape(1,-1)  # constraint on first moment
sigma2_constr = sigma2_x[step::step].reshape(1,-1)  # constraint on second moment

p, _ = PathMomMatch(q, X[:,step::step].T, mu_constr.T, sigma2_constr.T, p_constr.T)
# -

# ## Compute pdf to horizon

# +
option = namedtuple('option', 'n_bins')
option.n_bins = 100
[fx_hor, x_hor] = HistogramFP(X[:,[-1]].T, p[[-1],:], option)
[fy_hor, y_hor] = HistogramFP(Y[:,[-1]].T, p[[-1],:], option)

# stationary distribution of variance
y_stat = linspace(0, y_ + 2*eta, 2000)
fy_stat = gamma.pdf(y_stat, 2*kappa*y_/(eta**2), scale=eta**2 / (2*kappa))
# -

# ## Generate figure

# +
s_ = 252
# number of plotted observation before projecting time
j_visual = 10  # number of simulated paths to be printed
idx = range(j_visual)

# axes settings
m_x = min([npmin(X[idx, :]), npmin(x[:,-s_:]), mu_x[-1]-3.2*sqrt(sigma2_x[-1])])
M_x = max([npmax(X[idx, :]), npmax(x[:,-s_:]), mu_x[-1]+3.2*sqrt(sigma2_x[-1])])
m_y = min([npmin(Y[idx, :])*.91, npmin(y[:,-s_:])*.91, mu_y[-1] - 2.8*sqrt(sigma2_y[-1])])
M_y = max([npmax(Y[idx, :])*1.1, npmax(y[:,-s_:])*1.1, mu_y[-1] + 3.8*sqrt(sigma2_y[-1])])
tau_plot = tau / 252  #
t = arange(-1,tau_plot+1)  #
t_plot = t_j / 252  #
max_scale = tau_plot / 4  #

# preliminary computations
redline1_x = mu_x + 2*sqrt(sigma2_x)
redline2_x = mu_x - 2*sqrt(sigma2_x)
redline1_y = mu_y + 2*sqrt(sigma2_y)
redline2_y = mu_y - 2*sqrt(sigma2_y)

f, ax = subplots(2,1)
# color settings
lgrey = [0.8, 0.8, 0.8]
# light grey
dgrey = [0.55, 0.55, 0.55]
# dark grey
lblue = [0.27, 0.4, 0.9]
# light blue
# first subplot
plt.sca(ax[0])
plt.axis([t[0], t[-1] + max_scale, m_x, M_x])
xlabel('time (years)')
ylabel('Risk driver')
plt.grid(False)
title('Heston process')
# simulated paths
plot(t_plot, X[idx, :].T, color=lgrey, lw=2)
# standard deviation lines
p_red_1 = plot(t_plot, redline1_x, color='r', lw = 2)  # red bars (+2 std dev)
p_red_2 = plot(t_plot, redline2_x, color='r', lw = 2)  # red bars (-2std dev)
p_mu = plot(t_plot, mu_x, color='g', lw = 2)  # expectation
# histogram pdf plot
scale = max_scale / npmax(fx_hor)
for k in range(fx_hor.shape[1]):
    f_hist = plot([tau_plot, tau_plot+fx_hor[0,k]*scale], [x_hor[k], x_hor[k]], color=dgrey, lw=3)
# plot of last s_ observations
t_obs = arange(-1,1/252,1/252)
plot(t_obs, x[0,-s_-1:], color ='b',linestyle='none', marker = '.',markersize=1)

plot([tau_plot, tau_plot], mu_x[-1] + array([-2*sqrt(sigma2_x[-1]), + 2*sqrt(sigma2_x[-1])]), color='r', lw = 2)
# second subplot
plt.sca(ax[1])
plt.axis([t[0], t[-1] + max_scale, m_y, M_y])
xlabel('time (years)')
ylabel('Stochastic variance')
plt.grid(False)
title('Square-root (CIR) process')
# simulated paths
plot(t_plot, Y[idx, :].T, color=lgrey, lw=2)
# standard deviation lines
plot(t_plot, redline1_y, color='r', lw = 2)  # red bars (+2 std dev)
plot(t_plot, redline2_y, color='r', lw = 2)  # red bars (-2std dev)
plot(t_plot, mu_y, color='g', lw = 2)  # expectation
# histogram pdf plot
scale = max_scale / npmax(fy_hor)
for k in range(fy_hor.shape[1]):
    plot([tau_plot, tau_plot+fy_hor[0,k]*scale], [y_hor[k], y_hor[k]], color=dgrey, lw=3)
# stationary pdf
stationary = plot(tau_plot+fy_stat*scale, y_stat, color='k', lw=1)
# plot of last s_ observations
plot(t_obs, y[0,-s_-1:], color ='b',linestyle='none', marker = '.',markersize=1)
plot([tau_plot, tau_plot], mu_y[-1] + array([-2*sqrt(sigma2_y[-1]), + 2*sqrt(sigma2_y[-1])]), color='r', lw = 2)
# leg
leg = legend(handles=[p_mu[0], p_red_1[0], f_hist[0], stationary[0]],labels=['expectation', ' + / - 2st.deviation', 'horizon pdf','asymptotic distribution'])
plt.tight_layout();

# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

