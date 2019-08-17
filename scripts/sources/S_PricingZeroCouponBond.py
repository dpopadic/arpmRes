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

# # S_PricingZeroCouponBond [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_PricingZeroCouponBond&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-4-zcbvalue-evol).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import arange, array, ones, zeros, std, where, round, mean, log, tile, r_

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, xlim, scatter, ylabel, \
    xlabel, title, xticks

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot
from HistogramFP import HistogramFP
from SimVAR1MVOU import SimVAR1MVOU
from VAR1toMVOU import VAR1toMVOU
from FitVAR1 import FitVAR1
from InverseCallTransformation import InverseCallTransformation
from ZCBHorValue import ZCBHorValue
# -

# ## Upload the realized time series of the yield to maturities from db_SwapParRates.

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_SwapParRates'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_SwapParRates'), squeeze_me=True)

Rates = db['Rates']
Names = db['Names']
# -

# ## Select the key rates and compute the corresponding shadow rates
# ## using the last 1200 available observations with InverseCallTransformation.

t_end = 5
dt = 1 / 12
horiz_u = arange(0,t_end+dt,dt)
u_ = len(horiz_u)
timeStep = 1  # select time interval (days)
pick = arange(len(Names))  # select rates {'1y'[0] '2y'[1] '5y'[2] '7y'[3] '10y'[4] '15y'[5] '30y'[7-1]}
tau_d = array([[1, 2, 5, 7, 10, 15, 30]]).T
y = Rates[pick, 2001::timeStep]  # use the last 500 available observations
eta = 0.013
invcy = InverseCallTransformation(y, {1:eta})  # shadow rates

# ## Estimate the multivariate Ornstein-Uhlenbeck process parameters on the shadow rate time series using functions FitVAR1 and VAR1toMVOU.

# dinvcy = diff(invcy, 1, 2)
# [mu, theta, sigma2] = FitVAR1MVOU(dinvcy, invcy(:,1:-1), timeStep@1/252)
# [alpha, b, sig2_U] = FitVAR1(dinvcy, invcy(:,1:-1))
[alpha, b, sig2_U] = FitVAR1(invcy)
mu, theta, sigma2,*_ = VAR1toMVOU(alpha, b, sig2_U, timeStep*1 / 252)

# ## Project the multivariate Ornstein-Uhlenbeck process to future horizons by Monte Carlo method using function SimVAR1MVOU.

# +
j_ = 5000  # low for speed, increase for accuracy
x_0 = tile(invcy[:,[-1]], (1, j_))  # initial setup

X_u = SimVAR1MVOU(x_0, horiz_u[1:u_].reshape(1,-1), theta, mu, sigma2, j_)
X_u = r_['-1',x_0[...,np.newaxis], X_u]
# -

# ## Compute the zero-coupon bond value at future horizons using function ZCBHorValue,
# ##  along with the mean and the standard deviation of the zero-coupon bond.

# +
Z_u_t_end = zeros((1, j_, u_))
MuZ_u_t_end = zeros((1, 1, u_))
SigmaZ_u_t_end = zeros((1, 1, u_))

Z_u_t_end[0,:, 0] = ZCBHorValue(invcy[:,[-1]], tau_d, 0, t_end, 'shadow rates', {'eta':eta})
MuZ_u_t_end[0,0,0] = Z_u_t_end[0,0,0]
SigmaZ_u_t_end[0,0,0] = 0

for u in range(1, u_):
    Z_u_t_end[0,:, u]= ZCBHorValue(X_u[:,:, u], tau_d, horiz_u[u], t_end, 'shadow rates', {'eta':eta})
    MuZ_u_t_end[0,0, u] = mean(Z_u_t_end[0,:, u])
    SigmaZ_u_t_end[0,0, u] = std(Z_u_t_end[0,:, u])  # ## Compute the simulation probabilities and the average rates, then save the data in db_ZCB_value.

pp_ = ones((j_, 1)) / j_
MeanTenor_Rate = mean(X_u,axis=0)
# -

# ## Plot a few simulated paths of the zero-coupon bond value up to 3 years, along with the expectation, the standard deviation
# ##  and the horizon distribution. Furthermore represent the joint distribution of the zero-coupon bond value and the average rates at the selected horizon.

# +
lgrey = [0.8, 0.8, 0.8]  # light grey
dgrey = [0.4, 0.4, 0.4]  # dark grey
j_sel = 15  # selected MC simulations
hor_sel = 3  # selected horizon for the plot (10y)
i = where(horiz_u == hor_sel)[0][0]

figure()
# simulated path, mean and standard deviation
plot(horiz_u[:i+1].reshape(-1,1), Z_u_t_end[0, :j_sel, :i+1].T, color=lgrey,lw=1)
xticks(range(t_end))
xlim([min(horiz_u), max(horiz_u)+2])
l1 = plot(horiz_u[:i+1], MuZ_u_t_end[0, 0, :i+1], color='g')
l2 = plot(horiz_u[:i+1], MuZ_u_t_end[0, 0, :i+1] + SigmaZ_u_t_end[0, 0, :i+1], color='r')
plot(horiz_u[:i+1], MuZ_u_t_end[0, 0, :i+1] - SigmaZ_u_t_end[0, 0, :i+1], color='r')

# histogram
option = namedtuple('option', 'n_bins')
option.n_bins = round(10*log(j_))
y_hist, x_hist = HistogramFP(Z_u_t_end[:,:, i], pp_.T, option)
y_hist = y_hist*0.1  # adapt the hist height to the current xaxis scale
shift_y_hist = horiz_u[i] + y_hist

# empirical pdf
emp_pdf = plt.barh(x_hist[:-1], shift_y_hist[0]-horiz_u[i], left= horiz_u[i], height=x_hist[1]-x_hist[0],
                   facecolor=lgrey, edgecolor= lgrey)

# border
plot(shift_y_hist[0], x_hist[:-1], color=dgrey, lw=1)
legend(handles=[l1[0], l2[0], emp_pdf[0]],labels=['mean',' + / - st.deviation','horizon pdf'])
xlabel('time (years)')
ylabel('Normalized Value')
title('Zero-coupon projected value at the horizon');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

# scatter plot
figure()
scatter(MeanTenor_Rate[:,i], Z_u_t_end[0,:, i], 3, dgrey, '*')
xlabel('Average Rate')
ylabel('Normalized Value')
title('Scatter plot zero-coupon bond value vs. yield average');

# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
