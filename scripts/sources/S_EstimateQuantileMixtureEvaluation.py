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

# # S_EstimateQuantileMixtureEvaluation [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_EstimateQuantileMixtureEvaluation&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eSTaSSESSqUANT).

# ## Prepare the environment

# +
import os.path as path
import sys, os

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import ones, zeros, round, mean, log
from numpy import max as npmax
from numpy.random import rand

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, bar, legend, ylim, subplots, title

plt.style.use('seaborn')

from ARPM_utils import save_plot
from HistogramFP import HistogramFP
from QuantileMixture import QuantileMixture
# -

# ## Generate scenarios

# +
# define estimators
g_b = lambda x: mean(x, 1, keepdims=True)
g_e = lambda x: np.median(x, 1, keepdims=True)

# generate the scenarios for the time series
t_ = 50
j_ = 10 ** 3
alpha = 0.5
mu_Y = 0.1
sigma_Y = 0.2
mu_Z = 0
sigma_Z = 0.15

# compute the true value of the property
g_f = QuantileMixture(0.5, alpha, mu_Y, sigma_Y, mu_Z, sigma_Z)
# generate j_ simulations of the time series
I = zeros((j_, t_))
P = rand(j_, t_)
for j in range(j_):
    I[j,:] = QuantileMixture(P[j, :], alpha, mu_Y, sigma_Y, mu_Z, sigma_Z)
# -

# ## Compute error, bias and inefficiency for every estimator

# compute simulations of the estimators
G_b = g_b(I)
G_e = g_e(I)
# compute the losses of the estimators
L_b = (G_b - g_f) ** 2
L_e = (G_e - g_f) ** 2
# compute errors
er_b = mean(L_b)
er_e = mean(L_e)
# compute square bias
bias2_b = (mean((G_b) - g_f)) ** 2
bias2_e = (mean((G_e) - g_f)) ** 2
# compute square inefficiency
inef2_b = er_b - bias2_b
inef2_e = er_e - bias2_e

# ## Generate figures

# +
gray = [.7, .7, .7]
dgray = [.5, .5, .5]
red = [.9, .4, 0]
blue = [0, .45, .7]

# estimators.T distribution
f, ax = subplots(2,1)

NumBins = round(7*log(j_))
p = ones((1, j_)) / j_

plt.sca(ax[0])

option = namedtuple('option', 'n_bins')
option.n_bins = NumBins
n, x = HistogramFP(G_b.T, p, option)
b = bar(x[:-1], n[0], width=x[1]-x[0],facecolor=gray,edgecolor= dgray)
plot(g_f, 0, '.',markersize=15,color='g')
title('estimator a')
legend(['true property value'])

plt.sca(ax[1])

n, x = HistogramFP(G_e.T, p, option)
b = bar(x[:-1], n[0], width=x[1]-x[0], facecolor=gray,edgecolor= dgray)
plot(g_f, 0, '.',markersize= 15, color='g')
title('estimator b');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

# loss
f, ax = subplots(2,1)
h1 = 0.045
h = 0.01
plt.sca(ax[0])
n, x = HistogramFP(L_b.T, p, option)
b = bar(x[:-1], n[0], width=x[1]-x[0], facecolor=gray,edgecolor= dgray)
title('loss of estimator a')
ylim([0, npmax(n)*1.1])
bias = plot([0, bias2_b], [npmax(n)*h, npmax(n)*h], color=red, lw=3)
error = plot([0, er_b], [npmax(n)*h1, npmax(n)*h1], color='k',lw=3)
inefficiency = plot([bias2_b, er_b], [npmax(n)*h, npmax(n)*h], color=blue, lw=3)
legend(['error','bias$^2$' ,'ineff$^2$'])

plt.sca(ax[1])
n, x = HistogramFP(L_e.T, p, option)
b = bar(x[:-1], n[0], width=x[1]-x[0], facecolor=gray,edgecolor= dgray)
title('loss of estimator b')
ylim([0, npmax(n)*1.1])
plot([0, bias2_e], [npmax(n)*h, npmax(n)*h], color=red, lw=3)
plot([0, er_e], [npmax(n)*h1, npmax(n)*h1], color='k',lw=3)
plot([bias2_e, er_e], [npmax(n)*h, npmax(n)*h], color=blue, lw=3)
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
