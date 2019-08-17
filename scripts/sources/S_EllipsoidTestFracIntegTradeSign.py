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

# # S_EllipsoidTestFracIntegTradeSign [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_EllipsoidTestFracIntegTradeSign&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=IIDHFFIsign).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import min as npmin, max as npmax

from scipy.special import erf
from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict
from autocorrelation import autocorrelation
from FitFractionalIntegration import FitFractionalIntegration
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
index = ((bid!=ask) & (price == bid)) | ((price == ask) & (ask > bid))
frac = (price[index] - bid[index]) / (ask[index] - bid[index])
dz_k = erf(scale*(2*frac - 1))

# ## Fit the fractional integration process

# +
lag_ = 15  # max number of lags for the autocorrelation test

# trade sign as a fractional integration process
l_ = 50
d0 = 0

# epsFIsign are the residuals of a fractional integration process of order d+1
# computed as a sum truncated at order l_
# epsFIsign = (1-L)**(d+1) dz_k

d, epsFIsign, _, _, _ = FitFractionalIntegration(dz_k, l_, d0)
acf_epsFIsign = autocorrelation(epsFIsign, lag_)
# -

# ## Plot the results

# +
lag = 15  # lag to be printed
ell_scale = 1.7  # ellipsoid radius coefficient
fit = 0  # no fit on marginals
eps_lim = [npmin(epsFIsign), npmax(epsFIsign)]  # lim for the axes

f = figure(figsize=(14,7))
InvarianceTestEllipsoid(epsFIsign, acf_epsFIsign[0,1:], lag, fit, ell_scale, bound=eps_lim);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
plt.show()
