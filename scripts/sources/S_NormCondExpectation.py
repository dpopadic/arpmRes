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

# # S_NormCondExpectation [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_NormCondExpectation&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=EBCondExpNorm).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import array, ones, linspace, round, log, sqrt
from numpy import min as npmin, max as npmax

from scipy.stats import norm

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, bar, legend, xlim, scatter, ylabel, \
    xlabel

plt.style.use('seaborn')

from ARPM_utils import save_plot
from PlotTwoDimEllipsoid import PlotTwoDimEllipsoid
from HistogramFP import HistogramFP
from NormalScenarios import NormalScenarios

# inputs
j_ = 10000  # simulations
mu = array([[0.17], [- 2.5]])  # expectation
svec = array([[0.4], [1.2]])  # volatilities
rho = - 0.8  # correlation
s2 = np.diagflat(svec)@array([[1, rho], [rho, 1]])@np.diagflat(svec)  # covariance matrix
# -

# ## Generate bivariate normal simulations

Y,_ = NormalScenarios(mu, s2, j_, 'Riccati')
X = Y[[0]]
Z = Y[[1]]

# ## Compute the simulations of conditional expectation

phiZ = mu[0] + rho*svec[0] / svec[1]*(Z - mu[1])
mu_XphiZ = mu[0]*array([[1], [1]])  # joint expectation of X and E{X|Z}
pos = rho**2*s2[0, 0]
s2_XphiZ = array([[s2[0, 0], pos], [pos, pos]])  # covariance matrix of X and E{X|Z}

# ## Plot the empirical pdf of X and overlay the pdf of the conditional expectation

# +
nbins = round(7*log(j_))
figure()
p = ones((1, X.shape[1])) / X.shape[1]
option = namedtuple('option', 'n_bins')

option.n_bins = nbins
[n, x] = HistogramFP(X, p, option)
bar(x[:-1], n[0], width=x[1]-x[0], facecolor=[.8, .8, .8], edgecolor='k', label='empirical pdf of X')

pz_grid = linspace(npmin(x), npmax(x), 100)
f = norm.pdf(pz_grid, mu[0], sqrt(rho ** 2*s2[0, 0]))
plot(pz_grid, f, color='r', lw=2, label='analytical pdf of $E\{X | Z\}$')
xlim([min(x), npmax(x)])
legend();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
# -

# ## Display scatter plots of X and the conditional expectation, and their corresponding location-dispersion ellipsoids

# +
figure()
scatter(X, Z, 1, [.8, .8, .8], '*')
PlotTwoDimEllipsoid(mu, s2, 2, None, None, 'r', 2)
xlabel('X')
ylabel('Z');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

figure()
scatter(X, phiZ, 1, [.8, .8, .8], '*')
PlotTwoDimEllipsoid(mu_XphiZ, s2_XphiZ, 2, None, None, 'r', 2)
xlabel('X')
ylabel('$E\{X | Z\} = condexp_X(Z)$');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
