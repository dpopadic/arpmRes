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

# # S_EstimateMomentsMixtureEvaluation [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_EstimateMomentsMixtureEvaluation&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eSTaSSESSmOMbASED).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

from numpy import ones, zeros, round, mean, log, exp
from numpy import max as npmax
from numpy.random import rand

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, bar, legend, subplots, title

plt.style.use('seaborn')

from ARPM_utils import save_plot
from HistogramFP import HistogramFP
from QuantileMixture import QuantileMixture
# -

# ## Generate scenarios

# +
# define estimators
g_a = lambda X: (X[:, [0]] - X[:,[-1]]) *X[:, [1]] * X[:, [1]]
g_b = lambda X: mean(X, 1, keepdims=True)
g_c = lambda X: 5 + 0*X[:, [0]]
g_d = lambda X: mean(X ** 2 - X, 1, keepdims=True)

# generate the scenarios for the time series
t_ = 50
j_ = 10 ** 4
alpha = 0.5
mu_Y = 0.1
sigma_Y = 0.2
mu_Z = 0
sigma_Z = 0.15

# compute the true value of the property
gamma = alpha*(mu_Y ** 2 + sigma_Y ** 2 - mu_Y) + (1 - alpha)*( exp(2*mu_Z + 2*sigma_Z ** 2) - exp(mu_Z + 0.5*sigma_Z ** 2))
# generate j_ simulations of the time series
I = zeros((j_, t_))
P = rand(j_, t_)
for t in range(t_):
    I[:,t]= QuantileMixture(P[:,t], alpha, mu_Y, sigma_Y, mu_Z, sigma_Z)
# -

# ## Compute error, bias and inefficiency for every estimator

# compute simulations of the estimators
G_a = g_a(I)
G_b = g_b(I)
G_c = g_c(I)
G_d = g_d(I)
# compute the losses of the estimators
L_a = (G_a - gamma) ** 2
L_b = (G_b - gamma) ** 2
L_c = (G_c - gamma) ** 2
L_d = (G_d - gamma) ** 2
# compute errors
er_a = mean(L_a)
er_b = mean(L_b)
er_c = mean(L_c)
er_d = mean(L_d)
# compute square bias
bias2_a = (mean((G_a) - gamma)) ** 2
bias2_b = (mean((G_b) - gamma)) ** 2
bias2_c = (mean((G_c) - gamma)) ** 2
bias2_d = (mean((G_d) - gamma)) ** 2
# compute square inefficiency
inef2_a = er_a - bias2_a
inef2_b = er_b - bias2_b
inef2_c = er_c - bias2_c
inef2_d = er_d - bias2_d

# ## Generate figures

# +
gray = [.7, .7, .7]
dgray = [.5, .5, .5]
red = [.9, .4, 0]
blue = [0, .45, .7]

# estimators.T distributionfigure()
NumBins = round(7*log(j_))
p = ones((1, j_)) / j_

f, ax = subplots(4, 1)
plt.sca(ax[0])
option = namedtuple('option', 'n_bins')
option.n_bins = NumBins
n, x = HistogramFP(G_a.T, p, option)
b = bar(x[:-1],n[0], width=x[1]-x[0],facecolor=gray, edgecolor=dgray)

true = plot(gamma, 0, '.', markersize=15,color='g',label='true property value')
title('estimator a')
legend()

plt.sca(ax[1])
n, x = HistogramFP(G_b.T, p, option)
b = bar(x[:-1],n[0], width=x[1]-x[0],facecolor=gray, edgecolor=dgray)

plot(gamma, 0, '.',markersize= 15, color='g')
title('estimator b')

plt.sca(ax[2])
n, x = HistogramFP(G_c.T, p, option)
b = bar(x[:-1],n[0], width=x[1]-x[0],facecolor=gray, edgecolor=dgray)

plot(gamma, 0, '.',markersize= 15, color='g')
title('estimator c')

plt.sca(ax[3])
n, x = HistogramFP(G_d.T, p, option)
b = bar(x[:-1],n[0], width=x[1]-x[0],facecolor=gray, edgecolor=dgray)
plot(gamma, 0, '.', markersize=15, color='g')
title('estimator d')
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

# loss
h1 = 0.035
h = 0.01
f, ax = subplots(4,1)
plt.sca(ax[0])
n, x = HistogramFP(L_a.T, p, option)
b = bar(x[:-1],n[0], width=x[1]-x[0],facecolor=gray, edgecolor=dgray)
title('loss of estimator a')
plt.ylim([0,npmax(n)*1.1])
error = plot([0, er_a], [npmax(n)*h1, npmax(n)*h1], color='k',lw=2)
bias = plot([0, bias2_a], [npmax(n)*h, npmax(n)*h], color=red, lw=2)
inefficiency = plot([bias2_a, er_a], [npmax(n)*h, npmax(n)*h], color=blue, lw=2)
legend(['error', 'bias$^2$' , 'ineff$^2$'])

plt.sca(ax[1])
n, x = HistogramFP(L_b.T, p, option)
b = bar(x[:-1],n[0], width=x[1]-x[0],facecolor=gray, edgecolor=dgray)
title('loss of estimator b')
plt.ylim([0,npmax(n)*1.1])
plot([0, bias2_b], [npmax(n)*h, npmax(n)*h], color=red, lw=2)
plot([0, er_b], [npmax(n)*h1, npmax(n)*h1], color='k',lw=2)
plot([bias2_b, er_b], [npmax(n)*h, npmax(n)*h], color=blue, lw=2)

plt.sca(ax[2])
n, x = HistogramFP(L_c.T, p, option)
b = bar(x[:-1],n[0], width=x[1]-x[0],facecolor=gray, edgecolor=dgray)
title('loss of estimator c')
plt.ylim([0,npmax(n)*1.1])
plot([0, bias2_c], [npmax(n)*h, npmax(n)*h], color=red, lw=2)
plot([0, er_c], [npmax(n)*h1, npmax(n)*h1], color='k',lw=2)
plot([bias2_c, er_c], [npmax(n)*h, npmax(n)*h], color=blue, lw=2)

plt.sca(ax[3])
n, x = HistogramFP(L_d.T, p, option)
b = bar(x[:-1],n[0], width=x[1]-x[0],facecolor=gray, edgecolor=dgray)
title('loss of estimator d')
plt.ylim([0, npmax(n)*1.1])
plot([0, bias2_d], [npmax(n)*h, npmax(n)*h], color=red, lw=2)
plot([0, er_d], [npmax(n)*h1, npmax(n)*h1], color='k',lw=2)
plot([bias2_d, er_d], [npmax(n)*h, npmax(n)*h], color=blue, lw=2)
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
