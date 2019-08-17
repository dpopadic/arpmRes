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

# # S_NumAnEntropyGradComp [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_NumAnEntropyGradComp&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-comp-num-an-grad).

# ## Prepare the environment

# +
import os.path as path
import sys, os

from scipy.io import loadmat
from tqdm import trange

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

from numpy import ones, zeros, round, log
from numpy.random import randn
from numpy.linalg import norm

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, bar, title

plt.style.use('seaborn')

from ARPM_utils import save_plot
from HistogramFP import HistogramFP
from REnormLRD import REnormLRD
from numgrad import numgrad

# input parameters
j_ = 100  # number of simulations
n_ = 3  # market dimension
k_ = 2  # number of factors
# -

# ## Set random base-case parameters

# +
mu_ = randn(n_, 1)  # expectation
c = randn(n_, n_)
invs2_ = c@c.T  # inverse covariance

# relative entropy
e = lambda theta: REnormLRD(theta, mu_, invs2_, n_, k_)[0]
e2 = lambda theta: REnormLRD(theta, mu_, invs2_, n_, k_)[1]
# -

# ## Main computations

err = zeros((j_, 1))
for j in trange(j_,desc='Simulations'):
    # Set random variables
    theta_ = randn(n_ + n_*k_ + n_, 1)
    # Compute numerical gradient
    ngrad = numgrad(e, theta_)[0]
    ngrad = ngrad.flatten('F')
    # Compute analytical gradient
    angrad = e2(theta_)
    # Compute relative error
    err[j] = norm(angrad.flatten('F') - ngrad) / norm(angrad)

# ## Display the relative error

# +
nbins = int(round(10*log(j_)))
figure()

p = ones((1, len(err))) / len(err)
ax = plt.gca()
option = namedtuple('option', 'n_bins')
option.n_bins = nbins
[n, x] = HistogramFP(err.T, p, option)
b = bar(x[:-1], n[0],width=x[1]-x[0],facecolor= [.7, .7, .7])
plt.grid(True)
ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
title('Relative error');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

