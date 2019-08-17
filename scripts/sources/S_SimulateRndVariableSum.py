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

# # S_SimulateRndVariableSum [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_SimulateRndVariableSum&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=SumRndVarIndepPractice).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

from numpy import arange, ones, percentile, round, log, linspace, max as npmax, exp

from scipy.stats import chi2, expon, lognorm, gamma

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, bar, legend, title

plt.style.use('seaborn')

from ARPM_utils import save_plot
from HistogramFP import HistogramFP
# -

# ## Simulate Exponential(1/2) and Chi[3] distributions and compute their sum

# +
j_ = 100000  # len of samples

lam = 1 / 2  # exponential parameter (can't be changed)
nu = 4  # degrees of freedom for the Chi-squared distribution

Y = expon.rvs(scale=1 / lam, size=(1, j_))
Z = chi2.rvs(nu, size=(1, j_))

X = Y + Z
# -

# ## Compute the sample for a Gamma([6-1, 1-1]) distribution
# ## pay attention to the parameterizations

# +
min_x = 0
max_x = npmax(X) * 1.2
l_ = 2000
t = linspace(min_x, max_x, l_)

X_ = gamma.pdf(t, (nu + 2) / 2, scale=2)
# -

# ## Simulate a log-normal random variable R and compute the sample for T=R+Y

# +
mu = 3
sigma = 0.25

R = lognorm.rvs(sigma, scale=exp(mu),size=(1, j_))

T = R + Y
# -

# ## Create figures

# +
col = [0.94, 0.3, 0]
colhist = [.9, .9, .9]

# plot X
figure()
x_l = -0.1 * max_x
x_u = percentile(X, 100 * (1 - 10 ** -4))

p = ones((1, X.shape[1])) / X.shape[1]
option = namedtuple('option', 'n_bins')

option.n_bins = round(7 * log(j_))
hgram, xbins = HistogramFP(X, p, option)
h1 = bar(xbins[:-1], hgram[0], width=xbins[1]-xbins[0], facecolor=colhist, edgecolor='k')
plt.axis([x_l, x_u, 0, npmax(hgram) * 1.2])
h2 = plot(t, X_, color=col, lw=2)
title('Sum of random variables via simulation')
legend(['sum of exponential and chi-squared random var.', 'corresponding analytical gamma distribution']);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

# plot T
figure()
t_l = -0.1 * npmax(T)
t_u = percentile(T, 100 * (1 - 10 ** -4))
p = ones((1, T.shape[1])) / T.shape[1]
hgram2, xbins2 = HistogramFP(T, p, option)
h3 = bar(xbins2[:-1], hgram2[0], width=xbins2[1]-xbins2[0], facecolor=colhist, edgecolor='k')
plt.axis([t_l, t_u, 0, npmax(hgram2) * 1.2])
title('Sum of random variables via simulation')
legend(['sum of exponential and log-normal random variables']);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
