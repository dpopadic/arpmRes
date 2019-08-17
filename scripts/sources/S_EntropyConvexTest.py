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

# # S_EntropyConvexTest [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_EntropyConvexTest&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-test-convex).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

from numpy import ones, zeros, round, log, r_, array
from numpy.random import randn

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, bar, title

from tqdm import trange

plt.style.use('seaborn')

from ARPM_utils import save_plot, nullspace
from HistogramFP import HistogramFP
from REnormLRD import REnormLRD

# input parameters
j_ = 500  # number of simulations
n_ = 8  # market dimension
k_ = 3  # number of factors
# -

# ## Set random base-case parameters

mu_ = randn(n_, 1)  # expectation
c = randn(n_, n_)
invs2_ = c@c.T  # inverse covariance

# ## Main computations

# +
u = array([[0]])
for j in trange(j_, desc='Simulations'):  # Generate random coordinates
    theta_ = randn(n_*(2 + k_), 1)
    # Compute the relative entropy and a basis of the tangent space
    obj, grad, *_ = REnormLRD(theta_, mu_, invs2_, n_, k_)
    z = nullspace(grad.T)[1]
    # Compute the vector u
    m_ = n_*(2 + k_) - 1
    w = zeros((m_, 1))
    for m in range(m_):
        w[m],*_ = REnormLRD(theta_ + z[:, [m]], mu_, invs2_, n_, k_)[0][0] - obj[0]

    u = r_[u, w.copy()]
u = u[1:]
# ## Verify that u contains negative entries

nbins = round(10*log(j_))

figure()
p = ones((1, len(u))) / len(u)
option = namedtuple('option', 'n_bins')
option.n_bins = nbins
[n, x] = HistogramFP(u.T, p, option)
b = bar(x[:-1], n[0], width=x[1]-x[0], facecolor=[.7, .7, .7],edgecolor='k')
plt.grid(True)
title('Convex test');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
