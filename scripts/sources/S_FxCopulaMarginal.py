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

# # S_FxCopulaMarginal [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_FxCopulaMarginal&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-2-ex-fxcmfact).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

from numpy import ones, diff, round, log

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, bar, scatter, ylabel, \
    xlabel, title

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import struct_to_dict, save_plot
from HistogramFP import HistogramFP
from CopMargSep import CopMargSep
# -

# ## Load daily observations of the foreign exchange rates

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_FX'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_FX'), squeeze_me=True)

FXData = struct_to_dict(db['FXData'])

Y = FXData.Data
# select pair to spot
pair = [2, 3]  # 1 = Spot USD/EUR 2 = Spot USD/GBP 3 = Spot USD/JPY
# -

# ## Compute daily log-changes of the rates (Note: first column is time)

Epsi = diff(log(Y[:, 1:]), 1, 1)

# ## Compute FP-copula using the separation step of CMA

n_, t_ = Epsi.shape
p = ones((1, t_)) / t_  # flat Flexible Probabilities
_, _, U = CopMargSep(Epsi, p)

# ## Display the pdf of the copula of a normal distribution

# +
figure()
# empirical histograms of marginals
nbins = round(10*log(t_))
ax=plt.subplot2grid((3,3),(0,0), rowspan=2)
option = namedtuple('option', 'n_bins')
option.n_bins = nbins
[n, r] = HistogramFP(Epsi[[pair[1]],:], p, option)
plt.barh(r[:-1], n[0], height=r[1]-r[0], facecolor=[.8, .8, .8], edgecolor='none')

ax=plt.subplot2grid((3,3),(2,1), colspan=2)
[n, r] = HistogramFP(Epsi[[pair[0]],:], p, option)
bar(r[:-1], n[0], width=r[1]-r[0], facecolor=[.8, .8, .8], edgecolor=  'none')

# scatter plot
ax=plt.subplot2grid((3,3),(0,1), rowspan=2, colspan=2)
scatter(U[pair[0],:], U[pair[1],:], 0.5, [.5, .5, .5], '*')
title('Copula')
xlabel(str(FXData.Fields[pair[0]][0]))
ylabel(str(FXData.Fields[pair[1]][0]))
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])


