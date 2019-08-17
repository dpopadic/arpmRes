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

# # S_NumAnEntropyHessComp [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_NumAnEntropyHessComp&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-comp-num-an-hess).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

from scipy.io import loadmat
from scipy.linalg import kron
from tqdm import trange

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import ones, zeros, diag, eye, round, log, diagflat
from numpy.linalg import norm as linalgnorm
from numpy.random import randn

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, bar, title

plt.style.use('seaborn')

from ARPM_utils import save_plot
from HistogramFP import HistogramFP
from numHess import numHess
from REnormLRD import REnormLRD

# input parameters
j_ = 100  # number of simulations
n_ = 2  # market dimension
k_ = 1  # number of factors
# -

# ## Set random base-case parameters

# +
mu_ = randn(n_, 1)  # expectation
c = randn(n_, n_)

invs2_ = c@c.T  # inverse covariance

# set constant matrices for second derivatives
i_n = eye(n_)
i_k = eye(k_)

matrix = namedtuple('matrix', 'hm1 km')
matrix.hm1 = zeros((n_ ** 2, n_))
matrix.km = zeros((k_*n_, k_*n_))

for k in range(k_):
    matrix.km = matrix.km + kron(kron(i_k[:,[k]], i_n), i_k[:,[k]].T)

for n in range(n_):
    matrix.hm1 = matrix.hm1 + kron(i_n[:,[n]], diagflat(i_n[:,[n]]))  # relative entropy

e = lambda theta: REnormLRD(theta, mu_, invs2_, n_, k_, matrix)[0]
e3 = lambda theta: REnormLRD(theta, mu_, invs2_, n_, k_, matrix)[2]
# -

# ## Main computations

err = zeros((j_, 1))
for j in trange(j_,desc='Simulations'):
    # Set random variables
    theta_ = randn(n_ + n_*k_ + n_, 1)
    # Compute numerical Hessian
    numhess = numHess(e, theta_)[0]
    # Compute analytical Hessian
    anhess = e3(theta_)
    # Compute relative error in Frobenius norm
    err[j] = linalgnorm(anhess - numhess, ord='fro') / linalgnorm(anhess, ord='fro')

# ## Display the relative error

# +
nbins = int(round(10*log(j_)))
figure()

p = ones((1, len(err))) / len(err)
option = namedtuple('option', 'n_bins')

option.n_bins = nbins
[n, x] = HistogramFP(err.T, p, option)
b = bar(x[:-1], n[0], width=x[1]-x[0],facecolor= [.7, .7, .7])
plt.grid(True)
title('Relative error');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

