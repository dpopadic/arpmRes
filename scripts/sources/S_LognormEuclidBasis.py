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

# # S_LognormEuclidBasis [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_LognormEuclidBasis&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=EBRandGeomLogN).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import array, ones, diag, eye, abs, exp, sqrt

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, legend, xlim, ylim, scatter, subplots, ylabel, \
    xlabel, quiver

plt.style.use('seaborn')

from ARPM_utils import save_plot
from PlotTwoDimEllipsoid import PlotTwoDimEllipsoid
from NormalScenarios import NormalScenarios
from Riccati import Riccati

# input parameters
j_ = 5*10 ** 4  # number of simulations
m = array([[0.17], [0.06]])  # (normal) expectation
svec = array([[0.24], [0.14]])  # (normal) standard deviation
rho = 0.15  # (normal) correlation
# -

# ## Compute lognormal expectation and covariance

c2_ = array([[1, rho], [rho, 1]])  # (normal) correlation matrix
s2 = np.diagflat(svec)@c2_@np.diagflat(svec)  # (normal) covariance matrix
mu = exp(m + 0.5*diag(s2).reshape(-1,1))  # expectation
sig2 = np.diagflat(mu)@(exp(s2) - ones((2, 1)))@np.diagflat(mu)  # covariance matrix

# ## Generate bivariate lognormal draws

X = exp(NormalScenarios(m, s2, j_, 'Riccati')[0])

# ## Compute a the Riccati root of the correlation matrix and the vectors

# +
sigvec = sqrt(diag(sig2))  # standard deviation
c2 = np.diagflat(1 / sigvec)@sig2@np.diagflat(1 / sigvec)  # correlation matrix

c = Riccati(eye(2), c2)
x = c.T@np.diagflat(sigvec)
# -

# ## Compute Euclidean measures

inn_prods = x.T@x
lens = sqrt(diag(inn_prods))
angle = np.arccos(inn_prods[0, 1] / np.prod(lens))
distance = sqrt(inn_prods[0, 0] + inn_prods[1, 1] - 2*inn_prods[0, 1])

# ## Display the scatter plot and the ellipsoid

# +
x1 = max(abs((x[0])))
x2 = max(abs((x[1])))

f, ax = subplots(1,2)

plt.sca(ax[0])
scatter(X[0], X[1], 0.5, [.8, .8, .8], '*')
PlotTwoDimEllipsoid(mu, sig2, 1, [], 1, 'r', 2)
xlabel('$X_1$')
ylabel('$X_2$')
xlim([mu[0] - 1.5*x1, mu[0] + 1.5*x1])
ylim([mu[1] - 1.5*x2, mu[1] + 1.5*x2])
# -

# ## Display the vectors

plt.sca(ax[1])
quiver(0, 0, x[0, 0], x[1, 0], color = 'm', lw= 2, angles='xy',scale_units='xy',scale=1)
quiver(0, 0, x[0, 1], x[1, 1], color = 'b', lw= 2, angles='xy',scale_units='xy',scale=1)
quiv1 = plot(0, 0, color='m', lw= 2, marker=None)
quiv2 = plot(0, 0, color='b', lw= 2, marker=None)
plot(0, 0, 'o',markeredgecolor='k',markerfacecolor='w')
plt.grid(True)
xlim([- 1.5*x1, 1.5*x1])
ylim([- 1.5*x2, 1.5*x2])
legend(['$X_1$','$X_2$'])
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
