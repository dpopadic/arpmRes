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

# # S_NCopNMarginals [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_NCopNMarginals&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-2-ex-norm-cop-giv-norm-marg).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import array, ones, round, log, sqrt, r_

from scipy.stats import norm

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, bar, ylim, scatter, ylabel, \
    xlabel, title
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('seaborn')

from ARPM_utils import save_plot
from HistogramFP import HistogramFP
from NormalScenarios import NormalScenarios

# input parameters
j_ = int(1e4)  # number of simulations
rho = -0.8  # normal correlation
mu_X = array([[-2], [5]])  # normal expectation
svec_X = array([[1], [3]])  # normal standard deviations
# -

# ## Generate moment matching normal simulations

# +
c2_X = array([[1, rho], [rho, 1]])  # correlation matrix
s2_X = np.diagflat(svec_X)@c2_X@np.diagflat(svec_X)  # covariance matrix

X,_ = NormalScenarios(mu_X, s2_X, j_, 'Chol')
X_1 = X[0]
X_2 = X[1]
# -

# ## Compute the grades scenarios

U_1 = norm.cdf(X_1, mu_X[0], svec_X[0])  # grade 1
U_2 = norm.cdf(X_2, mu_X[1], svec_X[1])  # grade 2
U = r_[U_1, U_2]  # joint realizations from the required copula

# ## Scatter-plot of the marginals

figure()
scatter(X_1, X_2, 0.5, [.5, .5, .5], '*')
plt.grid(True)
xlabel('$X_1$')
ylabel('$X_2$')
title('Scatter plot of the marginals');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

# ## Scatter-plot of the grades

figure()
scatter(U_1, U_2, 0.5, [.5, .5, .5], '*')
plt.grid(True)
xlabel('grade U_1')
ylabel('grade U_2')
title('Scatter plot of the grades');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

# ## Histogram of the joint distribution

# +
f, ax = plt.subplots(1, 1, subplot_kw=dict(projection='3d'))
ax.view_init(34,-50)
NumBins2D = round(sqrt(100*log(j_)))
NumBins2D = array([[NumBins2D, NumBins2D]])

# set flat FP
p = ones((1, len(X[0]))) / len(X[0])
# compute histogram
option = namedtuple('option', 'n_bins')
option.n_bins = NumBins2D
[f, xi] = HistogramFP(X, p, option)
# plot histogram

xpos,ypos = np.meshgrid(xi[0][:-1], xi[1][:-1])
xpos = xpos.flatten('F')
ypos = ypos.flatten('F')
zpos = np.zeros_like(xpos)
# Construct arrays with the dimensions for the 16 bars.
dx = 0.5 * np.ones_like(zpos)
dy = dx.copy()
dz = f.flatten()
ax.bar3d(xpos,ypos,zpos, dx,dy,dz,color=[.8, .8, .8])
# ylim([min(xi[0, 0]), max(xi[0, 0])])
xlabel('$X_1$',labelpad=10)
ylabel('$X_2$',labelpad=10)
title('Histogram of the joint distribution');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
