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

# # S_NormEuclidBasis [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_NormEuclidBasis&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=EBRandGeomNorm).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import array, diag, eye, abs, sqrt
from numpy import max as npmax

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
mu = array([[0.67], [0.93]])  # expectation
svec = array([[2.14], [3.7]])  # standard deviation
rho = 0.47  # correlation
# -

# ## Generate bivariate normal draws

# +
c2 = array([[1, rho], [rho, 1]])  # correlation matrix
s2 = np.diagflat(svec)@c2@np.diagflat(svec)  # covariance matrix

X,_ = NormalScenarios(mu, s2, j_, 'Riccati')
# -

# ## Compute a the Riccati root of the correlation matrix and the vectors

c = Riccati(eye(2), c2)
x = c.T@np.diagflat(svec)

# ## Compute Euclidean measures

inn_prods = x.T@x
lens = sqrt(diag(inn_prods))
angle = inn_prods[0, 1] / np.prod(lens)

# ## Display the scatter plot and the ellipsoid

x1 = npmax(abs(x[0]))
x2 = npmax(abs(x[1]))
f, ax = subplots(1,2)
plt.sca(ax[0])
scatter(X[0], X[1], 0.5, [.8, .8, .8], '*')
PlotTwoDimEllipsoid(mu, s2, 1, [], 1, 'r', 2)
xlabel('$X_1$')
ylabel('$X_2$')
xlim([mu[0] - 1.5*x1, mu[0] + 1.5*x1])
ylim([mu[1] - 1.5*x2, mu[1] + 1.5*x2])

# ## Display the vectors

plt.sca(ax[1])
quiver(0, 0, x[0, 0], x[1, 0], color = 'm', lw= 2, angles='xy',scale_units='xy',scale=1)
quiver(0, 0, x[0, 1], x[1, 1], color = 'b', lw= 2, angles='xy',scale_units='xy',scale=1)
quiv1 = plot(0, 0, color='m', lw= 2)
quiv2 = plot(0, 0, color='b', lw= 2)
plot(0, 0, 'o',markeredgecolor='k',markerfacecolor='w')
plt.grid(True)
xlim([- 1.5*x1, 1.5*x1])
ylim([- 1.5*x2, 1.5*x2])
legend(['$X_1$','$X_2$']);
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
