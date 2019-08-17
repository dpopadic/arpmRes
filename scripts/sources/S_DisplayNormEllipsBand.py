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

# # S_DisplayNormEllipsBand [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_DisplayNormEllipsBand&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ProjCont).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import array, cos, sin, pi, linspace, diag, sqrt, r_

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, legend, scatter, title

plt.style.use('seaborn')

from ARPM_utils import save_plot
from PlotTwoDimEllipsoid import PlotTwoDimEllipsoid
from NormalScenarios import NormalScenarios
from PlotTwoDimBand import PlotTwoDimBand

# inputs
j_ = 10000
r = 3  # scale of the bands
n_points = 1000  # points of the bands
mu = array([[0.2], [0.5]])
sigma2 = array([[1, 0.5], [0.5, 0.8]])
# -

# ## Compute the standard deviations along the directions

# +
theta = linspace(0, 2*pi, n_points).reshape(1,-1)
u = r_[cos(theta), sin(theta)]  # directions

sigma_u = sqrt(diag(u.T@sigma2@u))  # projected standard deviations
# -

# ## Generate the normal sample

X,_ = NormalScenarios(mu, sigma2, j_, 'Chol')

# ## Display the band, the ellipsoid and overlay the scatterplot

figure(figsize=(10,10))
p1 = PlotTwoDimBand(mu, sigma_u, u, r, 'b')
p2 = PlotTwoDimEllipsoid(mu, sigma2, r, [], [], 'r')
scatter(X[0], X[1], s=5, c=[.3, .3, .3], marker='*')
legend(['Exp-Std. dev. band','Exp-Cov ellipsoid'])
title('Bivariate normal')
plt.axis('equal');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
