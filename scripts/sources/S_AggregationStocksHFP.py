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

# # S_AggregationStocksHFP [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_AggregationStocksHFP&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-stock-aggr-his-fp).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

from numpy import diff, round, log
from numpy.random import randint

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, title

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot
from HistogramFP import HistogramFP
# -

# ## Upload the database db_PricStocksHFP (computed in S_PricingStocksHFP)

try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_PricStocksHFP'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_PricStocksHFP'), squeeze_me=True)

# ## Generate a random vector of holdings h and aggregate the stocks P&L's scenarios
# ##  in the scenarios for the portfolio P&L

# +
Pi = db['Pi']
n_ = db['n_']
p = db['p'].reshape(1, -1)
ens = db['ens']

h = randint(-100, 100, (n_, 1))
Pi_h = h.T@Pi
# -

# ## Plot the histogram of the portfolio P&L Flexible Probabilities distribution at the horizon using function HistogramFP

figure()
opt = namedtuple('option', 'n_bins')
opt.n_bins = round(10 * log(ens))
f, x = HistogramFP(Pi_h, p, opt)
plt.bar(x[:-1], f[0], width=diff(x, 1)[0], facecolor=[.8, .8, .8], edgecolor=[.5, .5, .5])
title('Portfolio P&L Flexible Probabilities distribution');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
