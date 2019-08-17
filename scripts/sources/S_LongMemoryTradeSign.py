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

# # S_LongMemoryTradeSign [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_LongMemoryTradeSign&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=IIDHFLMsign).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import log
from numpy import max as npmax

from scipy.special import erf
from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, ylabel, \
    xlabel, title

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict
from autocorrelation import autocorrelation
from InvarianceTestEllipsoid import InvarianceTestEllipsoid

# -

# ## Upload database

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_StocksHighFreq'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_StocksHighFreq'), squeeze_me=True)

MSFT = struct_to_dict(db['MSFT'])

price = MSFT.trade_price
ask = MSFT.ask
bid = MSFT.bid
# -

# ## Compute the realizations of the variable "sign" (dz_k: difference in cumulative trade sign in tick time)
# ##take data with (ask > bid) and (price = bid or price = ask) and (bid different form ask)

scale = 5
index = (bid != ask) & (price == bid) | (price == ask) & (ask > bid)
frac = (price[index] - bid[index]) / (ask[index] - bid[index])
dz_k = erf(scale * (2 * frac - 1)).reshape(1, -1)

# ## Compute autocorrelations at different lags (needed for the invariance test)

lag_ = 15  # max number of lags for sign
acf_sign = autocorrelation(dz_k.reshape(1, -1), lag_)

# ## Compute the log-autocorrelations and perform a linear fit on the log-lags (to show the power decay)

# log-autocorrelations
lcr = log(acf_sign)
# linear fit
lag = range(1, lag_ + 1)
ll = log(lag)
p = np.polyfit(ll, lcr[0, 1:], 1)
y = p[0] * ll + p[1]

# ## Plot the results

# +
# settings
lag = 15  # lag to be printed
ell_scale = 1.8  # ellipsoid radius coefficient
fit = 0  # no fit on marginals
dz_k_lim = [-1.99, 1.99]  # lim for the axes
orange = [.9, .4, 0]

# autocorrelation test for invariance
f = figure(figsize=(12, 6))
InvarianceTestEllipsoid(dz_k, acf_sign[0, 1:], lag, fit, ell_scale, bound=dz_k_lim);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

# power low of autocorrelation decay
figure(figsize=(12, 6))
plot(ll, lcr[0, 1:], lw=1.5)
plot(ll, y, color=orange, lw=1.5)
plt.axis([min(ll), max(ll), min(lcr[0, 1:]), 0.95 * npmax(lcr[0, 1:])])
xlabel('ln(l)')
ylabel(r'$\ln( | Cr(\Delta\tilde\zeta_{\kappa},  \Delta\tilde\zeta_{\kappa-l}) | )$')
legend(['empirical', 'linear fit\n $\lambda$ =  % 1.3f' % -p[0]])
title('Autocorrelations decay: power law');
plt.show()
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
