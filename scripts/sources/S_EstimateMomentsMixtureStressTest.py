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

# # S_EstimateMomentsMixtureStressTest [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_EstimateMomentsMixtureStressTest&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=e-sta-ssessm-omb-ased-copy-1).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import arange, array, zeros, mean, exp
from numpy import min as npmin
from numpy.random import rand

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, bar, legend, ylim, subplots, title

plt.style.use('seaborn')

from ARPM_utils import save_plot
from QuantileMixture import QuantileMixture
# -

# ## Compute error, bias and inefficiency for every estimator and for every DGP within the stress-test set

# +
# define estimators
g_a =lambda X: (X[:, [0]] - X[:,[-1]]) *X[:, [1]] * X[:, [1]]
g_b =lambda X: mean(X, 1, keepdims=True)
g_c =lambda X: 5 + 0*X[:, [0]]
g_d =lambda X: mean(X ** 2 - X, 1, keepdims=True)

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
er_a = zeros(k_)
er_b = zeros(k_)
er_c = zeros(k_)
er_d = zeros(k_)
bias2_a = zeros(k_)
bias2_b = zeros(k_)
bias2_c = zeros(k_)
bias2_d = zeros(k_)
inef2_a = zeros(k_)
inef2_b = zeros(k_)
inef2_c = zeros(k_)
inef2_d = zeros(k_)
for k in range(k_):
    # compute the true value of the property
    mu_Y = S[k]
    g_f = alpha*(mu_Y ** 2+sigma_Y ** 2-mu_Y) + (1-alpha)*(exp(2*mu_Z+2*sigma_Z ** 2)-exp(mu_Z+0.5*sigma_Z ** 2))
    # generate j_ simulations of the time series
    P = rand(j_, t_)
    for t in range(t_):
        I[:,t] = QuantileMixture(P[:,t], alpha, mu_Y, sigma_Y, mu_Z, sigma_Z)

    # compute simulations of the estimators
    G_a = g_a(I)
    G_b = g_b(I)
    G_c = g_c(I)
    G_d = g_d(I)
    # compute the losses of the estimators
    L_a = (G_a - g_f) ** 2
    L_b = (G_b - g_f) ** 2
    L_c = (G_c - g_f) ** 2
    L_d = (G_d - g_f) ** 2
    # compute errors
    er_a[k] = mean(L_a)
    er_b[k] = mean(L_b)
    er_c[k] = mean(L_c)
    er_d[k] = mean(L_d)
    # compute square bias
    bias2_a[k] = (mean((G_a) - g_f)) ** 2
    bias2_b[k] = (mean((G_b) - g_f)) ** 2
    bias2_c[k] = (mean((G_c) - g_f)) ** 2
    bias2_d[k] = (mean((G_d) - g_f)) ** 2
    # compute square inefficiency
    inef2_a[k] = er_a[k] - bias2_a[k]
    inef2_b[k] = er_b[k] - bias2_b[k]
    inef2_c[k] = er_c[k] - bias2_c[k]
    inef2_d[k] = er_d[k] - bias2_d[k]
# -

# ## Compute robust and ensemble errors

# +
er_rob_a = max(er_a)
er_rob_b = max(er_b)
er_rob_c = max(er_c)
er_rob_d = max(er_d)

er_ens_a = mean(er_a)
er_ens_b = mean(er_b)
er_ens_c = mean(er_c)
er_ens_d = mean(er_d)
# -

# ## Determine the optimal estimator

# best robust estimator
er_rob, i_rob = npmin([er_rob_a, er_rob_b, er_rob_c, er_rob_d]), np.argmin([er_rob_a, er_rob_b, er_rob_c, er_rob_d])
# best ensemble estimator
er_ens, i_ens = npmin([er_ens_a, er_ens_b, er_ens_c, er_ens_d]), np.argmin([er_ens_a, er_ens_b, er_ens_c, er_ens_d])

# ## plot error, bias and inefficiency for each DGP within the stress-test set

# +
red = [.9, .4, 0]
blue = [0, .45, .7]

f, ax = subplots(4,1)

plt.sca(ax[0])
b = bar(range(1,len(bias2_a)+1), bias2_a+inef2_a, facecolor= red,label='bias$^2$')
b = bar(range(1,len(bias2_a)+1), inef2_a.T,facecolor= blue,label='ineff$^2$')
h = plot(range(1,len(bias2_a)+1), er_a, 'k',lw= 1.5,label='error')
yy = array(ylim())
plt.xticks(range(0,len(bias2_a)+1,2))
ylim(yy + array([0, 0.25]))
legend(frameon=True, ncol=3)
title('stress-test of estimator a')

plt.sca(ax[1])
b = bar(range(1,len(bias2_b)+1), bias2_b.T+inef2_b.T,facecolor= red)
b = bar(range(1,len(bias2_b)+1), inef2_b.T,facecolor= blue)
plot(range(1,len(bias2_b)+1), er_b, 'k',lw= 1.5)
plt.xticks(range(0,len(bias2_b)+1,2))
title('stress-test of estimator b')

plt.sca(ax[2])
b = bar(range(1,len(bias2_c)+1), bias2_c.T+inef2_c.T,facecolor= red)
b = bar(range(1,len(bias2_c)+1), inef2_c.T,facecolor= blue)
plot(range(1,len(bias2_c)+1), er_c, 'k',lw= 1.5)
plt.xticks(range(0,len(bias2_c)+1,2))
title('stress-test of estimator c')

plt.sca(ax[3])
b = bar(range(1,len(bias2_d)+1), bias2_d.T+inef2_d.T,facecolor= red)
b = bar(range(1,len(bias2_d)+1), inef2_d.T,facecolor= blue)
plot(range(1,len(bias2_d)+1), er_d, 'k',lw= 1.5)
plt.xticks(range(0,len(bias2_d)+1,2))
title('stress-test of estimator d');
plt.tight_layout();
plt.show()
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
