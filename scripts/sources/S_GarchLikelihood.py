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

# # S_GarchLikelihood [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_GarchLikelihood&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=exer-rel-mlesda-copy-2).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import arange, ones, var, array
from numpy import sum as npsum

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, ylabel, \
    xlabel

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict
from Price2AdjustedPrice import Price2AdjustedPrice
from FitGARCHFP import FitGARCHFP
# -

# ## Upload daily stock prices from db_Stocks

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_Stocks'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_Stocks'), squeeze_me=True)

StocksSPX = struct_to_dict(db['StocksSPX'])
# -

# ## Pick data for Apple, compute the compounded returns from dividend-adjusted stock prices

# +
_, dx = Price2AdjustedPrice(StocksSPX.Date.reshape(1,-1), StocksSPX.Prices[[0]], StocksSPX.Dividends[0])  # Apple returns
date = StocksSPX.Date[1:]

t_ = dx.shape[1]
# -

# ## GARCH(1,1) fit

# +
# initialize sigma**2 with a forward exponential smoothing
lam = 0.7
sig2_0 = lam*var(dx,ddof=1) + (1 - lam)*npsum((lam ** arange(1,t_+1)) * (dx ** 2))

# starting guess for the vector of parameters [c,a,b]
p0 = [0.7, .1, .2]

# constraint: a+b <= gamma
# gamma_grid=0.8:0.0range(1)
gamma_grid = arange(0.4,1.03,0.03)

# constant flexible probabilities
FP = ones((1, t_)) / t_

# fit
[par, _, _, lik] = FitGARCHFP(dx, sig2_0, p0, gamma_grid)
# -

# ## Figure

figure()
plot(gamma_grid, lik, lw=1.5)
ylabel('log-likelihood')
xlabel('$\gamma$(constraint: a + b $\leq$ $\gamma$)')
plt.xlim([min(gamma_grid),max(gamma_grid)]);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

