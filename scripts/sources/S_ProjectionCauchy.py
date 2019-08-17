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

# # S_ProjectionCauchy [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_ProjectionCauchy&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerCauchyProj).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import arange, array, linspace, log, exp, sqrt, r_
from numpy import sum as npsum, min as npmin, max as npmax

from scipy.stats import t
from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, scatter, ylabel, \
    xlabel, title

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict
from Price2AdjustedPrice import Price2AdjustedPrice
from MaxLikelihoodFPLocDispT import MaxLikelihoodFPLocDispT
from PathsCauchy import PathsCauchy
# -

# ## Upload databases

try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_Stocks'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_Stocks'), squeeze_me=True)

# ## Compute the log-returns of one stock

# +
StocksSPX = struct_to_dict(db['StocksSPX'])

x, dx = Price2AdjustedPrice(StocksSPX.Date.reshape(1,-1), StocksSPX.Prices[[25],:], StocksSPX.Dividends[25])  # Cisco Systems Inc
x = x[[0],1:]
t_ = x.shape[1]
# -

# ## Set the Flexible Probabilities

lam = log(2) / 800
p = exp((-lam * arange(t_, 1 + -1, -1))).reshape(1,-1)
p = p / npsum(p)  # FP-profile: exponential decay

# ## Fit the data to a Cauchy distribution

tol = 10 ** -6
nu = 1
mu, sigma2,_ = MaxLikelihoodFPLocDispT(dx, p, nu, tol, 1)
sigma = sqrt(sigma2)  # interquantile range corresponding to levels 1/4 and 3/4
mu = mu.squeeze()
sigma2 = sigma2.squeeze()
sigma = sigma.squeeze()

# ## Initialize projection variables

tau = 10
dt = 1 / 20
t_j = arange(0,tau+dt,dt)
j_ = 15

# ## Simulate paths

X = PathsCauchy(x[0,t_-1], mu, sigma, t_j, j_)

# ## Projection to horizon

# +
m_tau = x[0,t_-1] + mu*tau
sigma_tau = sigma*tau

# Cauchy pdf at horizon
l_ = 1000
x_hor = linspace(m_tau - 10 * sigma_tau,m_tau + 10*sigma_tau,l_)
# y_hor = t.pdf('tlocationscale', x_hor, m_tau, sigma_tau, 1)
y_hor = t.pdf((x_hor-m_tau)/sigma_tau,1)/sigma_tau
# -

# ## Create figure

# +
s_ = 2  # number of plotted observation before projecting time

m = min([npmin(X), npmin(x[0, t_ - s_: t_]), npmin([x[0,t_-1], m_tau]) - 6*sigma_tau])
M = max([npmax(X), npmax(x[0,t_ - s_: t_]), npmax([x[0,t_-1], m_tau]) + 6*sigma_tau])
t = arange(-s_,tau+1)
max_scale = tau / 4
scale = max_scale / max(y_hor)

# preliminary computations
tau_red = arange(0,tau,0.1).reshape(1,-1)
m_red = x[0,t_-1] + mu * tau_red
sigma_red = sigma * tau_red
redline1 = m_red + 2*sigma_red
redline2 = m_red - 2*sigma_red

f = figure()
# color settings
lgrey = [0.8, 0.8, 0.8]  # light grey
dgrey = [0.4, 0.4, 0.4]  # dark grey
lblue = [0.27, 0.4, 0.9]  # light blue
plt.axis([t[0], t[-1] + max_scale, m, M])
xlabel('time (days)')
ylabel('Risk driver')
plt.xticks(r_[t[:s_+1],arange(1,t[-1])])
title('Cauchy projection')
# simulated paths
for j in range(j_):
    plot(t_j, X[j,:], color = lgrey, lw = 2)
# standard deviation lines
p_red_1 = plot(tau_red[0], redline1[0], color='r', lw = 2,label=' + / - 2 sigma')  # red bars (+2 interquantile range)
p_red_2 = plot(tau_red[0], redline2[0], color='r', lw = 2)  # red bars (-2 interquantile range)
p_mu = plot([0, tau], [x[0,t_-1], m_tau], color='g', lw = 2, label='median')  # median
# histogram pdf plot
for k in range(len(y_hor)):
    p_hist=plot([tau, tau+y_hor[k]*scale], [x_hor[k], x_hor[k]], color=lgrey, lw=2,label='horizon pdf')
    plot(tau+y_hor*scale, x_hor, color=dgrey, lw=1)
# plot of last s_ observations
for k in range(s_):
    plot([t[k], t[k + 1]], [x[0,t_ - s_ + k - 1], x[0,t_ - s_ + k]], color=lgrey, lw=2)
for k in range(s_):
    scatter(t[k], x[0,t_ - s_ + k - 1], color='b',marker = '.',s=50)
    scatter(t[s_ + 1], x[0,t_-1], color='b', marker = '.',s=50)
plot([tau, tau], m_tau + array([-2 * sigma_tau, 2 * sigma_tau]), color='r', lw = 2)
# leg
legend(handles=[p_red_1[0], p_mu[0], p_hist[0]]);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
