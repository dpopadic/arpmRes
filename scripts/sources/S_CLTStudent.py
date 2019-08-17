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

# # S_CLTStudent [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_CLTStudent&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-2-ex-ind-vs-no-corr).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

from numpy import arange, ones, zeros, eye, round, log,sqrt, r_, min as npmin, max as npmax

from scipy.stats import norm, t

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, bar, legend, xlim, ylim, title

plt.style.use('seaborn')

from ARPM_utils import save_plot
from HistogramFP import HistogramFP
from Tscenarios import Tscenarios

# input parameters
n_ = 100  # number of variables
j_ = 5000  # number of simulations
nu = 5  # degrees of freedom
# -

# ## Generate iid t-draws

X_ = t.rvs(nu, size=(n_, j_))

# ## Generate uncorrelated t-draws

optionT = namedtuple('option', 'dim_red stoc_rep')
optionT.dim_red = 0
optionT.stoc_rep = 0
X = Tscenarios(nu, zeros((n_, 1)), eye(n_), j_, optionT, 'Chol')

# ## Compute the simulations of the sums

Y_ = ones((1, n_))@X_
Y = ones((1, n_))@X

# ## Plot normalized histograms and pdf's of the normal and t distributions

# +
nbins = round(12 * log(j_))

s = sqrt(n_ * (nu / (nu - 2)))

xmin = npmin(r_['-1',Y_, Y])
xmax = npmax(r_['-1',Y_, Y])

x = arange(xmin, xmax + 0.01, 0.01)

f = norm.pdf(x, 0, s)  # normal pdf

g = t.pdf(x / sqrt(n_), nu) / sqrt(n_)  # t pdf

fmax = npmax(r_['-1',f, g])

fig, ax = plt.subplots(2, 1)

# IID t-draws
plt.sca(ax[0])
p = ones((1, Y_.shape[1])) / Y_.shape[1]
option = namedtuple('option', 'n_bins')

option.n_bins = nbins
[n, y_] = HistogramFP(Y_, p, option)
bb = bar(y_[:-1], n[0], width=y_[1]-y_[0], facecolor=[.7, .7, .7],label='t-hist')

ff = plot(x, f, color='r', lw=2,label='normal pdf')
gg = plot(x, g, color='b', lw=2,label='t pdf')
xlim([xmin, xmax])
ylim([0, 1.3 * fmax])
legend()
title('IID t-draws')

# Uncorrelated t-draws
plt.sca(ax[1])
n, y = HistogramFP(Y, p, option)
bar(y[:-1], n[0], width=y[1]-y[0], facecolor=[.7, .7, .7])
plot(x, f, color='r', lw=2)
plot(x, g, color='b', lw=2)
xlim([xmin, xmax])
ylim([0, 1.3 * fmax])
title('Uncorrelated t-draws')
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
