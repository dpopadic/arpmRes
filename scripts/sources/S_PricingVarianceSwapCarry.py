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

# # S_PricingVarianceSwapCarry [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_PricingVarianceSwapCarry&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-4-carry-variance).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import ones, where, linspace, exp, min as npmin, max as npmax

from scipy.io import loadmat
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, subplots, ylabel, \
    xlabel, xticks

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict
from RollPrices2YieldToMat import RollPrices2YieldToMat

# initial settings
tau = 0.5  # t_end-t_start
upsilon = 1.5  # t_start-t
upsilon_u = linspace(upsilon, 0, 600)  # t_start-upsilon
# -

# ## Upload the realized time series of the rolling values and the (spot) variance spot rates on the S&P500

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_SwapCurve'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_SwapCurve'), squeeze_me=True)

DF_Rolling = struct_to_dict(db['DF_Rolling'])

try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_VarianceSwap'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_VarianceSwap'), squeeze_me=True)

VarianceSwapRate = struct_to_dict(db['VarianceSwapRate'])
# -

# ## Select today rolling prices, where today is t= 03-Oct-2012. Compute the yield with time to maturity upsilon+tau needed to compute the vega carry

y, _ = RollPrices2YieldToMat(DF_Rolling.TimeToMat[DF_Rolling.TimeToMat == 2],
                             DF_Rolling.Prices[DF_Rolling.TimeToMat == 2, DF_Rolling.Dates == VarianceSwapRate.date].reshape(1,-1))

# ## Compute the forward variance swap rate  sigma2_{t}(upsilon,tau)

forwardVariance = ((upsilon + tau) / tau)*VarianceSwapRate.SwapRate[VarianceSwapRate.timeToMat == upsilon + tau]\
                  -(upsilon / tau)*VarianceSwapRate.SwapRate[VarianceSwapRate.timeToMat == upsilon]
# ForwardVariance((upsilon,tau))

# ## After interpolating the values of the spot variance swap rates sigma2_{t}(upsilon_u) and  sigma2_{t}(upsilon_u+tau), compute the forward variance swap rate at the horizon  sigma2_{t}(upsilon_u,tau)

# +
# Spot variance swap rates
interp = interp1d(VarianceSwapRate.timeToMat, VarianceSwapRate.SwapRate, fill_value='extrapolate')
spotVariance2 = interp(upsilon_u)

# spotvariance((upsilon_u))
spotVariance1 = interp(upsilon_u + tau)

# spotvariance((upsilon_u+tau))

# Forward variance swap rate at horizon

forwardVarianceHor = ((upsilon_u + tau) / tau) * spotVariance1 - (upsilon_u / tau) * spotVariance2
# -

# ## Compute the "vega" carry of the forward start variance swap

vegaCarry = exp((-(upsilon + tau))*y)*(forwardVarianceHor - forwardVariance)

# ## Plot the vega carry at a selected horizon (upsilon = 1 year),
# ## along with the forward variance swap rate curve corresponding to the steady path as a function to the time to start at the horizon (upsilon_u).

# +
f, ax = subplots(2,1)

mC = npmin(vegaCarry)
MC = npmax(vegaCarry)
k_ = len(upsilon_u)
time_to_u = upsilon - upsilon_u
grey = [.7, .7, .7]  # light grey
xtl = [' 0','3m', ' 6m', ' 9m', ' 1y', '15m', '18m']
k = where(upsilon_u > 0.5)[0][-1]

# "vega" carry curve
plt.sca(ax[0])
xlabel('Time to horizon')
ylabel('Vega carry')
p1 = plot([time_to_u[k_ - k + 1], time_to_u[k_ - k + 1]], [mC, MC], color='k',lw= 2)
plt.fill_between(time_to_u[k_ - k :k_],vegaCarry[0,:k],0,facecolor= grey, edgecolor= grey)
plt.axis([time_to_u[0], time_to_u[-1], mC, MC])
xticks(time_to_u[k_ - k :k_:120],xtl)
plt.grid(True)

# Forward variance swap rate at horizon curve

plt.sca(ax[1])
xlabel('Time between horizon and start')
ylabel('Forward variance swap rate')
plot(upsilon_u, forwardVarianceHor, color='b',lw= 2)
plot([upsilon_u[k], upsilon_u[k]],[forwardVarianceHor[k], forwardVarianceHor[k]], color = 'r',
marker='.',markersize= 15)
plt.axis([upsilon_u[-1], upsilon_u[0], min(forwardVarianceHor), max(forwardVarianceHor)])
plt.grid(True)
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

