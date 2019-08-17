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

# # S_EllipsoidTestPnLGARCH [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_EllipsoidTestPnLGARCH&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ex-p-and-lres-ell-test).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import abs

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot
from autocorrelation import autocorrelation
from InvarianceTestEllipsoid import InvarianceTestEllipsoid
from GarchResiduals import GarchResiduals
# -

# ## Upload database

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_MomStratPL'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_MomStratPL'), squeeze_me=True)

dailypnl = db['dailypnl']
# -

# ## Select data

pi = dailypnl.reshape(1,-1)  # select observations

# ## Compute the invariants using GARCH(1,1) fit and test invariance

epsi = GarchResiduals(pi)  # GARCH fit

# ## Perform autocorrelation test on y and epsi

# +
lag_ = 10  # number of lags (for auto correlation test)

acf_y = autocorrelation(abs(pi), lag_)
acf_epsi = autocorrelation(abs(epsi), lag_)
# -

# ## Generate figures

# +
lag = 10  # lag to be printed
ell_scale = 2  # ellipsoid radius coefficient
fit = 0  # normal fitting

# axis settings
rpi = np.ptp(abs(pi))
repsi = np.ptp(abs(epsi))

# position settings
pos = {}
pos[0] = [.2, .45, .3866, .43]  # scatter plot
pos[1] = [.2905, .12, .205, .2157]  # epsi
pos[2] = [.01, .45, .1737, .43]  # epsi_lagged
pos[3] = [.6, .45, .3366, .43]  # autocorrelation
pos[4] = [.085, .228, .11, .1]  # leg

f = figure(figsize=(12,6))
InvarianceTestEllipsoid(abs(pi), acf_y[0,1:], lag, fit, ell_scale, pos, 'P&L', [-rpi / 8, 0]);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

f = figure(figsize=(12,6))  # changes in log implied vol
InvarianceTestEllipsoid(abs(epsi), acf_epsi[0,1:], lag, fit, ell_scale, [], 'GARCH residuals', [-repsi / 8, 0]);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
