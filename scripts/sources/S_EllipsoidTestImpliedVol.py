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

# # S_EllipsoidTestImpliedVol [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_EllipsoidTestImpliedVol&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=IIDtestImpliedVol).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import reshape, ones, diff, eye, log, r_
from numpy.linalg import solve

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot
from autocorrelation import autocorrelation
from InvarianceTestEllipsoid import InvarianceTestEllipsoid
# -

# ## Upload database

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_Derivatives'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_Derivatives'), squeeze_me=True)

Sigma = db['Sigma']
# -

# ## Select weekly observations of implied volatility

delta_t = 5
sigma = Sigma[:,:, ::delta_t]

# ## Compute autocorrelation, at different lags, of weekly changes in implied vol

# +
tau_index = 1  # time to maturity index
m_index = 4  # moneyness index
lag_ = 10  # total number of lags

delta_sigma = diff(sigma[tau_index, [m_index],:])  # changes in implied volatility
acf_1 = autocorrelation(delta_sigma, lag_)
# -

# ## Compute autocorrelation, at different lags, of weekly changes in log implied vol

# +
log_sigma = log(sigma[tau_index, [m_index],:])  # logarithm of implied vol

delta_log_sigma = diff(log_sigma)  # changes in log implied volatility
acf_2 = autocorrelation(delta_log_sigma, lag_)
# -

# ## Perform the least squares fitting and compute autocorrelation of residuals

# +
tau_, m_, t_ = sigma.shape
sigma = reshape(sigma, (tau_*m_, t_),'F')

y = sigma[:, 1:].T
x = r_['-1',ones((t_ - 1, 1)), sigma[:, : -1].T]

yx = y.T@x
xx = x.T@x
b = yx@(solve(xx,eye(xx.shape[0])))
r = y - x@b.T  # residuals

epsi = r[:, [2]].T  # select the residuals corresponding to 60 days-to-maturiy and moneyness equal to 0.9
acf_3 = autocorrelation(epsi, lag_)
# -

# ## Plot the results of the IID test

# +
lag = 10  # lag to be printed
ell_scale = 2  # ellipsoid radius coefficient
fit = 0  # normal fitting

f = figure(figsize=(14,7))  # changes in implied vol
InvarianceTestEllipsoid(delta_sigma,acf_1[0,1:], lag, fit, ell_scale, [], 'IID test on the increments of implied volatility');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

f = figure(figsize=(14,7))  # changes in log implied vol
InvarianceTestEllipsoid(delta_log_sigma,acf_2[0,1:], lag, fit, ell_scale, [], 'IID test on the increments of log implied volatility');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

f = figure(figsize=(14,7))  # residuals of the autoregressive fit
InvarianceTestEllipsoid(epsi,acf_3[0,1:], lag, fit, ell_scale, [], 'IID test on the residuals of the autoregressive fit');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

