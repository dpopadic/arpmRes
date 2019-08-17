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

# # S_DisplayNormWishMargEllipsBand [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_DisplayNormWishMargEllipsBand&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=EllipsBandNormWishMarg).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import reshape, trace, array, zeros, cos, sin, pi, linspace, \
    diag, sqrt, r_
from numpy.linalg import det
from numpy.random import multivariate_normal as mvnrnd

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, legend, scatter, ylabel, \
    xlabel, title

plt.style.use('seaborn')

from ARPM_utils import save_plot
from PlotTwoDimEllipsoid import PlotTwoDimEllipsoid
from PlotTwoDimBand import PlotTwoDimBand

# input parameters
sigvec = array([[1], [1]])  # dispersion parameters
rho = -0.9  # correlation parameter
nu = 5  # deegrees of freedom
j_ = 10000  # number of simulations
n_points = 1000  # points of the uncertainty band
r = 3  # radius of the ellipsoid
# -

# ## Generate simulations

# +
W_11 = zeros((1, j_))
W_22 = zeros((1, j_))
W_12 = zeros((1, j_))
vec_W = zeros((4, j_))
dets = zeros((1, j_))
traces = zeros((1, j_))

sig2 = np.diagflat(sigvec) @ array([[1, rho], [rho, 1]]) @ np.diagflat(sigvec)

for j in range(j_):
    X = mvnrnd(zeros(2), sig2, nu).T
    W = X @ X.T

    dets[0, j] = det(W)
    traces[0, j] = trace(W)

    W_11[0, j] = W[0, 0]
    W_22[0, j] = W[1, 1]
    W_12[0, j] = W[0, 1]

    vec_W[:, [j]] = reshape(W, (4, 1))

# expected values of W_11 and W_12
E_11 = nu * sig2[0, 0]
E_12 = nu * sig2[0, 1]

# covariance matrix of W_11 and W_12
V_11 = nu * (sig2[0, 0] * sig2[0, 0] + sig2[0, 0] * sig2[0, 0])
V_12 = nu * (sig2[0, 0] * sig2[1, 1] + sig2[0, 1] * sig2[1, 0])
Cv_11_12 = nu * (sig2[0, 0] * sig2[0, 1] + sig2[0, 1] * sig2[0, 0])

Cv_W11_W12 = array([[V_11, Cv_11_12], [Cv_11_12, V_12]])
# -

# ## Compute normalized variables X_1 and X_2

# +
X_1 = (W_11 - E_11) / sqrt(V_11)
X_2 = (W_12 - E_12) / sqrt(V_12)
X = r_[X_1, X_2]

# expected value and covariance of (X_1, X_2)
E_X = array([[0], [0]])
Sd_W11_W12 = array([[sqrt(V_11)], [sqrt(V_12)]])
Cv_X = np.diagflat(1 / Sd_W11_W12) @ Cv_W11_W12 @ np.diagflat(1 / Sd_W11_W12)
# -

# ## Compute the standard deviations along the directions

# +
theta = linspace(0, 2 * pi, n_points).reshape(1, -1)
u = r_[cos(theta), sin(theta)]  # directions

s_u = sqrt(diag(u.T @ Cv_X @ u))  # projected standard deviations
# -

# ## Display the band, the ellipsoid and overlay the scatterplot

# +
figure(figsize=(10, 10))

p1 = PlotTwoDimBand(E_X, s_u, u, r, 'b')
p2 = PlotTwoDimEllipsoid(E_X, Cv_X, r, [], [], 'r')
scatter(X[0], X[1], s=5, c=[.3, .3, .3], marker='*')
legend(['Mean-Cov band', 'Mean-Cov ellipsoid'])
title('Normalized Wishart marginals')
xlabel('$X_1$')
ylabel('$X_2$')
plt.axis('equal');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

