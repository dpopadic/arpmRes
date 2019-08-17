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

# # S_EstimateQuantileMixtureStressTest [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_EstimateQuantileMixtureStressTest&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=e-sta-ssessq-uant-copy-1).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import arange, zeros, var, \
    mean
from numpy.random import rand

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, bar, legend, subplots, title

plt.style.use('seaborn')

from ARPM_utils import save_plot
from QuantileMixture import QuantileMixture
# -

# ## Compute error, bias and inefficiency for every estimator and for every DGP within the stress-test set

# +
# define estimators
g_b = lambda X: mean(X, 1, keepdims=True)
g_e = lambda X: np.median(X, 1, keepdims=True)

# generate the scenarios for the time series
t_ = 50
j_ = 10 ** 4
alpha = 0.5
sigma_Y = 0.2
mu_Z = 0
sigma_Z = 0.15

S = arange(0,0.22,0.02)  # stress-test set for parameter mu_Y
k_ = len(S)

I = zeros((j_, t_))
er_b = zeros(k_)
er_e = zeros(k_)
bias2_b = zeros(k_)
bias2_e = zeros(k_)
inef2_b = zeros(k_)
inef2_e = zeros(k_)
for k in range(k_):
    # compute the true value of the property
    mu_Y = S[k]
    g_f = QuantileMixture(0.5, alpha, mu_Y, sigma_Y, mu_Z, sigma_Z)
    # generate j_ simulations of the time series
    P = rand(j_, t_)
    for j in range(j_):
        I[j,:] = QuantileMixture(P[j, :], alpha, mu_Y, sigma_Y, mu_Z, sigma_Z)

    # compute simulations of the estimators
    G_b = g_b(I)
    G_e = g_e(I)
    # compute the losses of the estimators
    L_b = (G_b - g_f) ** 2
    L_e = (G_e - g_f) ** 2
    # compute errors
    er_b[k] = mean(L_b)
    er_e[k] = mean(L_e)
    # compute square bias
    bias2_b[k] = (mean((G_b) - g_f)) ** 2
    bias2_e[k] = (mean((G_e) - g_f)) ** 2
    # compute square inefficiency
    inef2_b[k] = var(G_b, ddof=1)
    inef2_e[k] = var(G_e, ddof=1)
# -

# ## Compute robust and ensemble errors

# +
er_rob_b = max(er_b)
er_rob_e = max(er_e)

er_ens_b = mean(er_b)
er_ens_e = mean(er_e)
# -

# ## Determine the optimal estimator

# best robust estimator
er_rob = min([er_rob_b, er_rob_e]),
# best ensemble estimator
er_ens = min([er_ens_b, er_ens_e])

# ## plot error, bias and inefficiency for each DGP within the stress-test set

# +
red = [.9, .4, 0]
blue = [0, .45, .7]

f, ax = subplots(2,1)
plt.sca(ax[0])
b = bar(range(1,k_+1),bias2_b.T+inef2_b.T, facecolor= red, label='bias$^2$')
b = bar(range(1,k_+1),inef2_b.T,facecolor= blue,label='ineff$^2$')
plot(range(1,k_+1), er_b, 'k',lw=1.5, label='error')
plt.xticks(range(0,k_+2,2))
legend()
title('stress-test of estimator b')

plt.sca(ax[1])
b = bar(range(1,k_+1),bias2_e.T+inef2_e.T,facecolor= red)
b = bar(range(1,k_+1),inef2_e.T,facecolor= blue)
plot(range(1,k_+1), er_e, 'k',lw= 1.5)
plt.xticks(range(0,k_+2,2))
title('stress-test of estimator e')
plt.tight_layout();
plt.show()
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
