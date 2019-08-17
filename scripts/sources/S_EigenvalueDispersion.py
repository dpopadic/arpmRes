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

# # S_EigenvalueDispersion [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_EigenvalueDispersion&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerEigDisp).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import arange, zeros, sort, argsort, cov, eye
from numpy.linalg import eig
from numpy.random import multivariate_normal as mvnrnd

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, subplots, ylabel, \
    xlabel
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('seaborn')

from ARPM_utils import save_plot

# inputs
i_ = 50  # dimension of the covariance matrix
t_vector = i_ * arange(1, 11)  # different lens of the time series
j_ = 50  # simulations for each time series
mu = zeros(i_)
sigma2 = eye(i_)
# -

# ## Compute sample eigenvalues from time series of different lens

lambda2_hat = zeros((len(t_vector), i_))
for k in range(len(t_vector)):
    t_ = t_vector[k]
    lambda2_tmp = 0
    for j in range(j_):
        # simulate the time series
        Epsi = mvnrnd(mu, sigma2, t_).T
        # compute sample covariance
        sigma2_hat = cov(Epsi, ddof=1)
        # compute eigenvalues
        l, _ = eig(sigma2_hat)
        l, Index = sort(l)[::-1], argsort(l)[::-1]
        lambda2_tmp = lambda2_tmp + l

    # average of eigenvalues across different scenarios
    lambda2_tmp = lambda2_tmp / j_
    # store the resulting average eigenvalues
    lambda2_hat[k, :] = lambda2_tmp

# ## Create figure

# Display surface
x, y = np.meshgrid(range(i_), t_vector / i_)
f, ax = subplots(1, 1, subplot_kw=dict(projection='3d'))
ax.view_init(30,-120)
ax.plot_surface(x, y, lambda2_hat)
xlabel('eigenvalue #',labelpad=10)
ylabel('sample length/i',labelpad=10)
plt.grid(True);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
