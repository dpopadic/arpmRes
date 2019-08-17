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

# # S_IncludeDataMVEConvergence [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_IncludeDataMVEConvergence&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerMVEConvergence).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import r_

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, xlim, ylim, scatter, ylabel, \
    xlabel

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict
from PlotTwoDimEllipsoid import PlotTwoDimEllipsoid
from Price2AdjustedPrice import Price2AdjustedPrice
from GarchResiduals import GarchResiduals
from IncludeDataMVE import IncludeDataMVE
# -

# ## Upload the database

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_Stocks'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_Stocks'), squeeze_me=True)

StocksSPX = struct_to_dict(db['StocksSPX'])
# -

# ## Compute the dividend-adjusted returns of two stocks

# +
n_ = 2
t_ = 750

_, x_1 = Price2AdjustedPrice(StocksSPX.Date.reshape(1,-1), StocksSPX.Prices[[25],:], StocksSPX.Dividends[25])  # Cisco Systems Inc
_, x_2 = Price2AdjustedPrice(StocksSPX.Date.reshape(1,-1), StocksSPX.Prices[[5],:], StocksSPX.Dividends[5])  # General Electric
date = StocksSPX.Date[1:]

x_1 = x_1[[0],-t_:]
x_2 = x_2[[0],-t_:]
date = date[-t_:]
# -

# ## Compute the invariants using GARCH(1,1) fit
epsi = GarchResiduals(r_[x_1,x_2])

# +
# ## Compute mean and covariance at each iterative step of the algorithm

mu, sigma2, bound = IncludeDataMVE(epsi)
# -

# ## Generate a figure showing the ellipsoids computed at each iterative step of the algorithm.

# +
k_ = mu.shape[1]
c_thin = [1, 0.5, 0.5]

Xlim = [min(epsi[0]) - 1, max(epsi[0])+1]
Ylim = [min(epsi[1]) - 1, max(epsi[1])+1]

figure()

scatter(epsi[0], epsi[1], 3, 'b', '*')
xlabel('$\epsilon_1$')
ylabel('$\epsilon_2$')
xlim(Xlim)
ylim(Ylim)

for k in range(k_ - 1):
    PlotTwoDimEllipsoid(mu[:,[k]], sigma2[:,:,k], 1, 0, 0, c_thin, 1.2)
PlotTwoDimEllipsoid(mu[:,[k_-1]], sigma2[:,:, k_-1], 1, 0, 0, 'r', 2)
iterT = 'Number of Iterations:  % 3.0f'%k_
plt.text(Xlim[0] + 0.5, Ylim[1] + 0.1, iterT, color='k',horizontalalignment='left',verticalalignment='bottom')

scatter(epsi[0, bound], epsi[1, bound], 15, 'b')
scatter(epsi[0, bound], epsi[1, bound], 40, 'k');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
