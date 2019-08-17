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

# # S_DisplayAlternativeBands [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_DisplayAlternativeBands&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=VisuUncertBands).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import array, zeros, cos, sin, pi, linspace, diag, sqrt, r_

from scipy.stats import norm

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, scatter, title

plt.style.use('seaborn')

from ARPM_utils import save_plot
from NormalScenarios import NormalScenarios
from PlotTwoDimBand import PlotTwoDimBand

# inputs
j_ = 10000
r = 2  # scale of the bands
n_points = 1000  # points of the bands
mu = array([[0.7], [0.5]])
sigma2 = array([[2, -1], [- 1, 1.5]])
# -

# ## Compute locations and dispersions along the directions

# +
theta = linspace(0, 2*pi, n_points).reshape(1,-1)
u = r_[cos(theta), sin(theta)]  # directions

mu_u = u.T@mu  # projected expectations
sigma_u = sqrt(diag(u.T@sigma2@u)).reshape(-1,1)  # projected standard deviations
median_u = norm.ppf(0.5, mu_u, sigma_u)  # projected medians
range_u = norm.ppf(0.75, mu_u, sigma_u) - norm.ppf(0.25, mu_u, sigma_u)  # projected ranges
# -

# ## Compute the alternative location-dispersion bands

band_points1 = zeros((2, n_points))
band_points2 = zeros((2, n_points))
for n in range(n_points):
    band_points1[:,n] = (mu_u[n] + r*sigma_u[n])*u[:,n]
    band_points2[:,n] = (median_u[n] + r*range_u[n])*u[:,n]

# ## Generate the normal sample

X,_ = NormalScenarios(mu, sigma2, j_, 'Chol')

# ## Display the bands and overlay the scatterplot

# +
figure()

p1 = PlotTwoDimBand(mu, sigma_u, u, r, 'r')
p2 = plot(band_points1[0], band_points1[1], color='b', lw=2)
scatter(X[0], X[1], s=5, c=[.3, .3, .3], marker='*')
legend(['Band','Alternative Band'])
title('Expectation-Std Deviation bands of a bivariate normal');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

figure()
pp1 = PlotTwoDimBand(mu, range_u, u, r, 'r')
pp2 = plot(band_points2[0], band_points2[1], color='b', lw=2)
scatter(X[0], X[1], s=5, c=[.3, .3, .3], marker='*')
legend(['Band','Alternative Band'])
title('Median-Range bands of a bivariate normal');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
