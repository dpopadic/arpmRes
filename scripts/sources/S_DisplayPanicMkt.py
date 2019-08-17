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

# # S_DisplayPanicMkt [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_DisplayPanicMkt&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-2-panic-mark).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

from numpy import arange, diff, round, log, corrcoef

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, bar, scatter, ylabel, \
    xlabel, title

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict
from HistogramFP import HistogramFP
from PanicTDistribution import PanicTDistribution
from CopMargSep import CopMargSep
from ColorCodedFP import ColorCodedFP

# inputs
j_ = 1000  # number of simulations
nb = round(5*log(j_))

nu = 3  # degree of freedom
r = 0.85  # panic correlation
c = 0.15  # threshold
# -

# ## Load daily observations of the stocks in S&P 500

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_StocksS_P'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_StocksS_P'), squeeze_me=True)

Data = struct_to_dict(db['Data'])
V = Data.Prices
pair = [0, 1]  # stocks to spot
# -

# ## Set the calm correlation matrix as sample correlation matrix of compounded returns

# +
C = diff(log(V), 1, 1)
C = C[pair, :]

varrho2 = corrcoef(C)
# -

# ## Compute panic distribution

X, p_ = PanicTDistribution(varrho2, r, c, nu, j_)

# ## Extract the simulations of the panic copula

x, u, U = CopMargSep(X, p_)

# ## Represent the scatter-plot of panic distribution plot the histograms of their marginals

# +
# scatter plot
figure()
grey_range = arange(0,0.81,0.01)
CM, C = ColorCodedFP(p_, None, None, grey_range, 0, 18, [17, 5])
# colormap(CM)
scatter(X[0], X[1], s=3, c=C, marker='.',cmap=CM)
xlabel('$X_1$')
ylabel('$X_2$')
title('Panic joint distribution');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

# marginal X1
figure()
option = namedtuple('option', 'n_bins')
option.n_bins = nb
f, c1 = HistogramFP(X[[0]], p_, option)
bar(c1[:-1], f[0], width=c1[1]-c1[0], facecolor=[.9, .9, .9], edgecolor=  'k')
title('Marginal $X_1$');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

# marginal X2
figure()

f, varrho2 = HistogramFP(X[[1]], p_, option)
bar(varrho2[:-1], f[0], width=varrho2[1]-varrho2[0], facecolor=[.9, .9, .9], edgecolor=  'k')
title('Marginal $X_2$');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
# -

# ## Scatter-plot the simulations of panic copula U and plot the histograms of the grades

# +
# scatter plot
figure()

grey_range = arange(0,0.81,0.01)
CM, C = ColorCodedFP(p_, None, None, grey_range, 0, 18, [17, 5])
# colormap(CM)
scatter(U[0], U[1], s=3, c=C, marker='.',cmap=CM)
xlabel('$U_1$')
ylabel('$U_2$')
title('Panic copula');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

# grade U1
figure()
[f, c1] = HistogramFP(U[[0]], p_, option)
bar(c1[:-1], f[0],  width=c1[1]-c1[0], facecolor=[.9, .9, .9], edgecolor=  'k')
title('Grade $U_1$');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

# Grade U2
figure()
f, varrho2 = HistogramFP(U[[1]], p_, option)
bar(varrho2[:-1], f[0], width=varrho2[1]-varrho2[0], facecolor=[.9, .9, .9], edgecolor=  'k')
title('Grade $U_2$');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

