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

# # S_NumAnSigNoGradComp [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_NumAnSigNoGradComp&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-comp-num-an-sig-no-grad).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

from scipy.io import loadmat
from tqdm import trange

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

from numpy import ones, zeros, diag, eye, round, log, array
from numpy.linalg import norm
from numpy.random import randn

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, bar, title

plt.style.use('seaborn')

from ARPM_utils import save_plot
from HistogramFP import HistogramFP
from SigNoConstrLRD import SigNoConstrLRD
from numjacobian import numjacobian

# input parameters
j_ = 100  # number of simulations
n_ = 3  # market dimension
k_ = 2  # number of factors
m_ = 1  # number of constraints
# -

# ## Specify the constraint function with random parameters

# +
a = randn(m_, n_)
q = randn(m_, 1)

# set constant matrices for derivatives
i_n = eye(n_)
matrix = namedtuple('matrix','hm hm2 km1')
matrix.hm = diag(i_n.flatten())
matrix.hm2 = array([])
matrix.km1 = array([])

# constraint function
v = lambda theta: SigNoConstrLRD(theta, a, q, n_, k_, matrix)[0]
v2 = lambda theta: SigNoConstrLRD(theta, a, q, n_, k_, matrix)[1]
# -

# ## Main computations

err = zeros((j_, 1))
for j in trange(j_,desc='Simulations'):
    # Set random variables
    theta_ = randn(n_ + n_*k_ + n_, 1)
    # Compute numerical gradient
    ngrad = numjacobian(v, theta_)[0]
    ngrad = ngrad.T
    # Compute analytical gradient
    angrad = v2(theta_).reshape(-1,1)
    # Compute relative error in Frobenius norm
    err[j] = norm(angrad - ngrad, ord='fro') / norm(angrad, ord='fro')

# ## Display the relative error

# +
nbins = round(10*log(j_))
figure()

p = ones((1, len(err))) / len(err)
option = namedtuple('option', 'n_bins')

option.n_bins = nbins
ax = plt.gca()
[n, x] = HistogramFP(err.T, p, option)
b = bar(x[:-1], n[0], width=x[1]-x[0], facecolor= [.7, .7, .7])
ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
plt.grid(True)
title('Relative error');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
