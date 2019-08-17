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

# # S_ExecutionAlmgrenChrissMultidim [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_ExecutionAlmgrenChrissMultidim&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-plopt_-liquidation-trajectories_m-ac-copy-1).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import arange, array, zeros, sinh, diag, eye, sqrt, r_
from numpy.linalg import eig, solve

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, ylabel, \
    xlabel, title

plt.style.use('seaborn')

from ARPM_utils import save_plot

# input parameters
n_ = 3  # number of traded assets
h_0 = array([[150, 110, 85]])  # initial holdings
g = array([[1.32, 0, 0], [0, 1.22, 0], [0, 0, 1.4]])  # temporary impact matrix
sigma = array([[1, 0.2, 0.3], [0.4, 1, 0.8], [0.5, 0.2, 1]])  # variance-covariance matrix
sigma2 = sigma@sigma.T
lam = 2  # risk aversion coefficient
q_end = 1
m_ = 100  # total number of discrete trades
epsilon = q_end / m_  # discretization step
# -

# ## Compute the matrix sigma2_tilde and the matrix a appearing in the system of finite differences

inv_g = solve(sqrt(g),eye(g.shape[0]))
sigma2_tilde = sigma2 + diag(diag(sigma2))
a = lam*0.5*inv_g@sigma2_tilde@inv_g

# ## Compute the numerical solution of the multidimensional Almgren-Chriss model

# +
lambda_tilde, u = eig(a)
lambda_tilde = lambda_tilde.reshape(-1,1)  # eigenvalues of matrix a
lambda_sign = np.arccosh(epsilon ** 2*lambda_tilde / 2 + 1) / epsilon
z_0 = u.T@sqrt(g)@h_0.T
z = zeros((n_, m_))
h = zeros((n_, m_))

for m in range(m_):
    z[:,[m]] = (sinh(lambda_sign*(q_end - (m+1)*epsilon)) / sinh(lambda_sign*q_end)) * z_0
    h[:, m] = inv_g@u@z[:, m]
# -

# ## Plot the trading trajectories of the three assets.

# +
figure()

q_grid = arange(0,q_end+epsilon,epsilon)
p1 = plot(q_grid, r_[h_0[0,0], h[0]], color='b', marker = '.',markersize=5,lw=1)
p2 = plot(q_grid, r_[h_0[0,1], h[1]], color='r', marker = '.',markersize=5,lw=1)
p3 = plot(q_grid, r_[h_0[0,2], h[2]], color ='k', marker = '.',markersize=5,lw=1)

xlabel('Volume time')
ylabel('Share holdings')
title('Optimal trajectories in the multidimensional Almgren-Chriss model')
legend(['first asset','second asset','third asset']);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
