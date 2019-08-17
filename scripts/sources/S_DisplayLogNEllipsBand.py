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

# # S_DisplayLogNEllipsBand [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_DisplayLogNEllipsBand&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=EllipsBandLogNorm).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import array, zeros, cos, sin, pi, percentile, linspace, diag, exp, r_

import matplotlib.pyplot as plt
from matplotlib.pyplot import legend, scatter, title

plt.style.use('seaborn')

from ARPM_utils import save_plot
from NormalScenarios import NormalScenarios
from PlotTwoDimBand import PlotTwoDimBand
from PlotTwoDimEllipsoid import PlotTwoDimEllipsoid

# inputs
j_ = 10000
r = 3  # scale of the bands
n_points = 1000  # points of the bands
mu = array([[0.3],[0.1]])
sigma2 = array([[0.06, -0.03], [- 0.03, 0.02]])
# -

# ## Compute expectation and covariance

m = exp(mu.flatten() + 0.5*diag(sigma2))
s2 = np.diagflat(m)@(exp(sigma2) - 1)@np.diagflat(m)

# ## Generate the lognormal sample

# +
Norm, _ = NormalScenarios(mu, sigma2, j_, 'Chol')

X = exp(Norm)
# -

# ## Compute medians and interquantile ranges along the directions

# +
theta = linspace(0, 2*pi, n_points).reshape(1,-1)
u = r_[cos(theta), sin(theta)]

# projected medians
med = zeros((2,1))
med[0] = percentile((array([[1,0]])@X).T, 50)
med[1] = percentile((array([[0,1]])@X).T, 50)

# projected interquantile ranges
range_u = zeros((n_points, 1))
for n in range(n_points):
    range_u[n] = percentile((u[:, n].T@X).T, 75) - percentile((u[:, n].T@X).T, 25)  # warning: if slow decrease n_points
# -

# ## Display the band, the ellipsoid and overlay the scatterplot

p1 = PlotTwoDimBand(med, range_u, u, r, 'b')
p2 = PlotTwoDimEllipsoid(m[...,np.newaxis], s2, r, False, False, 'r')
p3 = scatter(X[0], X[1], color= [.3, .3, .3], marker='*',s=0.5)
legend(['Median-Range band','Mean-Cov ellipsoid'])
title('Bivariate lognormal');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
