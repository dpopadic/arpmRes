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

# # S_ProjectionAutocorrelatedProcess [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_ProjectionAutocorrelatedProcess&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-sim-ouprocess).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import arange, ones, zeros, where, squeeze, \
    linspace, round, log, sqrt, tile, r_
from numpy import min as npmin, max as npmax
from numpy.linalg import solve

from scipy.stats import norm
from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, xlim, ylim, title, xticks

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict
from HistogramFP import HistogramFP
from RollPrices2YieldToMat import RollPrices2YieldToMat
from ProjMomentsVAR1MVOU import ProjMomentsVAR1MVOU
from SimVAR1MVOU import SimVAR1MVOU
from VAR1toMVOU import VAR1toMVOU
from FitVAR1 import FitVAR1
# -

# ## Upload the realized time series of the rolling values contained in db_SwapCurve.

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_SwapCurve'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_SwapCurve'), squeeze_me=True)

DF_Rolling = struct_to_dict(db['DF_Rolling'])
# -

# ## Select the five-year rolling prices using the last 1000 available observations
# ## and compute the corresponding yields to maturity using function RollPrices2YieldToMat.

# horizon
t_end = 15
dt = 0.1
horiz_u = arange(0,t_end+dt,dt)
u_ = len(horiz_u)
timeStep = 5  # select frequency (days)
tau = 5  # selected maturity (5y)
prices = DF_Rolling.Prices[DF_Rolling.TimeToMat == tau, 2221:] # use the last 1500 available observations
x = RollPrices2YieldToMat(tau, prices)[0]

# ## Estimate the Ornstein-Uhlenbeck process parameters using functions FitVAR1 and VAR1toMVOU.

# dx = diff(x, 1, 2)
# [mu, theta, sigma2] = FitVAR1MVOU(dx, x[:-1], timeStep/252)
# [alpha, b, sig2_U] = FitVAR1(dx, x[:-1])
[alpha, b, sig2_U] = FitVAR1(x)
mu, theta, sigma2,*_ = VAR1toMVOU(alpha, b, sig2_U, timeStep / 252)

# ## Project the Ornstein-Uhlenbeck process to future horizons by Monte Carlo method using function SimVAR1MVOU
# ## and compute the simulation probabilities.

j_ = 3000  # increase for accuracy
x_0 = tile(x[0,[-1]], (1, j_))  # initial setup
X = SimVAR1MVOU(x_0, horiz_u[1:].reshape(1,-1), theta, mu.reshape(-1,1), sigma2, j_)
X = r_['-1',x_0[...,np.newaxis], X[np.newaxis,...]]
pp_ = ones((j_, 1)) / j_  # simulation probabilities

# ## Compute and plot the projected distribution using function ProjMomentsVAR1MVOU, the Brownian approximation
# ## and the stationary distribution at the selected horizons (6 months and 13 years).
# ## Show also a few simulated paths, along with the mean and the standard deviation of the projected distrubution.

# +
hor_sel1 = 0.5  # selected horizon (6 months)
i1 = where(horiz_u == hor_sel1)[0][0]
hor_sel2 = 13  # selected horizon (13y)
i2 = where(horiz_u == hor_sel2)[0][0]

for i in [i1, i2]:
    t_end = 500
    x_Hor = zeros((1, t_end, u_))
    y_Hor = zeros((1, t_end, u_))
    y_Hor_brownian = zeros((1, t_end, u_))
    y_Hor_asympt = zeros((1, t_end, u_))

    # parameters of exact distribution
    [mu_u, sigma2_u, drift_u] = ProjMomentsVAR1MVOU(x[0,[-1]], horiz_u, mu.reshape(-1,1), theta, sigma2)
    sigma_u = squeeze((sqrt(sigma2_u))).T
    # parameters of Brownian motion approximation
    exp_brown = x[0,-1] + mu*horiz_u
    sigma_brown = sqrt(sigma2*horiz_u)
    # parameters of asymptoptic approximation
    exp_asympt = mu / theta
    sigma_asympt = sqrt(solve(2*theta,sigma2))

    x_Hor[0,:,i] = linspace(drift_u[0,i] - 20*sqrt(sigma2_u[0, i]), drift_u[0,i] + 20*sqrt(sigma2_u[0, i]),t_end)
    y_Hor[0,:,i] = norm.pdf(x_Hor[0,:,i], drift_u[0,i], sigma_u[i])  # Analytical projection at horizon
    y_Hor_brownian[0,:,i] = norm.pdf(x_Hor[0,:,i], exp_brown[i], sigma_brown[0,i])  # Brownian approximation
    y_Hor_asympt[0,:,i] = norm.pdf(x_Hor[0,:,i], exp_asympt, sigma_asympt)  # Normal asymptoptic approximation

    # figure

    lgrey = [0.8, 0.8, 0.8]  # light grey
    dgrey = [0.4, 0.4, 0.4]  # dark grey
    lblue = [0.27, 0.4, 0.9]  # light blu
    j_sel = 15  # selected MC simulations

    figure()

    # simulated path, mean and standard deviation
    plot(horiz_u[:i], X[0, :j_sel, :i].T, color=lgrey)
    xticks(range(15))
    xlim([npmin(horiz_u) - 0.01, 17])
    ylim([-0.03, 0.06])
    l1 = plot(horiz_u[:i], x[0,-1] + mu_u[0, :i], color='g',label='Expectation')
    l2 = plot(horiz_u[:i], x[0,-1] + mu_u[0, :i] + sigma_u[:i], color='r', label=' + / - st.deviation')
    plot(horiz_u[:i], x[0,-1] + mu_u[0, :i] - sigma_u[:i], color='r')

    # analytical pdf
    option = namedtuple('option', 'n_bins')
    option.n_bins = round(10*log(j_))
    y_hist, x_hist = HistogramFP(X[[0],:,i], pp_.T, option)
    scale = 200*sigma_u[i] / npmax(y_hist)
    y_hist = y_hist*scale
    shift_y_hist = horiz_u[i] + y_hist

    emp_pdf = plt.barh(x_hist[:-1], shift_y_hist[0]-horiz_u[i], height=x_hist[1]-x_hist[0],
                       left=horiz_u[i], facecolor=lgrey, edgecolor= lgrey, lw=2,label='Horizon pdf')
    plot(shift_y_hist[0], x_hist[:-1], color=dgrey)  # border

    # Brownian approximation
    y_Hor_brownian[0,:,i] = y_Hor_brownian[0,:,i]*scale
    shift_y_brown = zeros(y_Hor_brownian.shape)
    shift_y_brown[0,:,i] = horiz_u[i] + y_Hor_brownian[0,:,i]
    l4 = plot(shift_y_brown[0,:,i], x_Hor[0,:,i], color = lblue, label='Brownian approx')

    # asymptotic approximation
    y_Hor_asympt[0,:, i] = y_Hor_asympt[0,:, i]*scale
    shift_y_asympt = zeros(y_Hor_asympt.shape)
    shift_y_asympt[0,:, i] = horiz_u[i] + y_Hor_asympt[0,:, i]
    l5 = plot(shift_y_asympt[0,:, i], x_Hor[0,:, i], color = dgrey, label='Asymptotic distribution')
    legend()
    title('Ornstein-Uhlenbeck process');
    # save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

