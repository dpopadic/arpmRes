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

# # S_SemicircleDistribution [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_SemicircleDistribution&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerRandomMatrix).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

from numpy import arange, ones, pi, ceil, log, exp, sqrt, linspace
from numpy.linalg import eig, eigvals
from numpy.random import rand, randn

from scipy.stats import expon, lognorm

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, bar, legend, title

plt.style.use('seaborn')

from ARPM_utils import save_plot
from HistogramFP import HistogramFP

# initialize variables
i_ = 1000  # matrix size
# -

# ## Generate matrix Y

X_1 = randn(i_,i_)  # standard normal invariants
Y_1 = (X_1 + X_1.T) / (2*sqrt(2*i_))
X_2 = expon.rvs(scale=1, size=(i_,i_)) - 1  # exponential invariants
Y_2 = (X_2 + X_2.T) / (2*sqrt(2*i_))
X_3 = (rand(i_,i_) - 0.5)*sqrt(12)  # uniform invariants
Y_3 = (X_3 + X_3.T) / (2*sqrt(2*i_))
X_4 = (lognorm.rvs(1, scale=1, size=(i_,i_))- exp(0.5)) / sqrt(exp(2) - exp(1))  # log-normal distribution
Y_4 = (X_4 + X_4.T) / (2*sqrt(2*i_))

# ## Compute the sample eigenvalues and the corresponding normalized histograms

# +
nbins = int(ceil(10*log(i_)))
option = namedtuple('option', 'n_bins')

option.n_bins = nbins

# standard normal
Lambda2_1 = eigvals(Y_1)
p_flat = ones((1, len(Lambda2_1))) / len(Lambda2_1)
hgram_1, x_1 = HistogramFP(Lambda2_1.reshape(1,-1), p_flat, option)
# exponential
Lambda2_2 = eigvals(Y_2)
hgram_2, x_2 = HistogramFP(Lambda2_2.reshape(1,-1), p_flat, option)
# uniform
Lambda2_3 = eigvals(Y_3)
hgram_3, x_3 = HistogramFP(Lambda2_3.reshape(1,-1), p_flat, option)
# log-normal
Lambda2_4 = eigvals(Y_4)
hgram_4, x_4 = HistogramFP(Lambda2_4.reshape(1,-1), p_flat, option)
# -

# ## Compute the semicircle function

# +
x = linspace(-1,1,200)

g = 2 / pi*sqrt(1 - x ** 2)
# -

# ## Create figures

# +
figure()
bar(x_1[:-1], hgram_1[0], width=x_1[1]-x_1[0], facecolor= [.7, .7, .7], edgecolor= [.5, .5, .5])
plot(x, g, 'r',lw= 2)
title('Standard Normal variables')
legend(['Sample eigenvalues','Semicircle function']);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

# exponential
figure()
bar(x_2[:-1], hgram_2[0], width=x_2[1]-x_2[0],facecolor= [.7, .7, .7], edgecolor= [.5, .5, .5])
plot(x, g, 'r',lw= 2)
title('Exponential variables')
legend(['Sample eigenvalues','Semicircle function']);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

# uniform
figure()
bar(x_3[:-1], hgram_3[0], width=x_3[1]-x_3[0],facecolor= [.7, .7, .7], edgecolor= [.5, .5, .5])
plot(x, g, 'r',lw= 2)
title('Uniform variables')
legend(['Sample eigenvalues','Semicircle function']);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

# log-normal
figure()
bar(x_4[:-1], hgram_4[0], width=x_4[1]-x_4[0],facecolor= [.7, .7, .7], edgecolor= [.5, .5, .5])
plot(x, g, 'r',lw= 2)
title('Log-normal variables')
legend(['Sample eigenvalues','Semicircle function']);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
