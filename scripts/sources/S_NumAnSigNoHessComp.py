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

# # S_NumAnSigNoHessComp [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_NumAnSigNoHessComp&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-comp-num-an-sig-no-hess).

# ## Prepare the environment

# +
import os.path as path
import sys, os

from scipy.io import loadmat

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import ones, zeros, diag, eye, round, log, r_, diagflat
from numpy.linalg import norm as linalgnorm
from numpy.random import randn
from scipy.linalg import kron
from tqdm import trange

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, bar, title

plt.style.use('seaborn')

from ARPM_utils import save_plot
from HistogramFP import HistogramFP
from SigNoConstrLRD import SigNoConstrLRD
from numHess import numHess

# input parameters
j_ = 100  # number of simulations
n_ = 2  # market dimension
k_ = 1  # number of factors
m_ = 1  # number of constraints
# -

# ## Specify the constraint function with random parameters

# +
a = randn(m_, n_)
q = randn(m_, 1)

# set constant matrices for derivatives
i_n = eye(n_)
i_k = eye(k_)

matrix = namedtuple('matrix', 'hm km1 hm2')
matrix.hm = diag(i_n.flatten())
matrix.km1 = zeros((k_*n_, k_*n_ ** 2))
matrix.hm2 = zeros((n_, n_ ** 2))
for n in range(n_):
    matrix.hm2 = matrix.hm2 + kron(i_n[:,[n]].T, diagflat(i_n[:,[n]]))
    matrix.km1 = matrix.km1 + kron(kron(i_n[:,[n]].T, i_k), diagflat(i_n[:,[n]]))  # constraint function

v =lambda theta: SigNoConstrLRD(theta, a, q, n_, k_, matrix)[0]
v3 =lambda theta: SigNoConstrLRD(theta, a, q, n_, k_, matrix)[2]
# -

# ## Main computations

err = zeros((j_, 1))
for j in trange(j_,desc='Simulations'):  # Set random variables
    theta_ = randn(n_ + n_*k_ + n_, 1)
    # Compute numerical Hessian
    for m in range(m_):
        g_m =lambda theta: SigNoConstrLRD(theta, a[[m],:], q[m], n_, k_)[0]
        h = numHess(g_m, theta_)[0]
        if m==0:
            numhess=h.copy()
        else:
            numhess = r_['-1',numhess, h]  # Compute analytical Hessian
    anhess = v3(theta_)
    # Compute relative error in Frobenius norm
    err[j] = linalgnorm(anhess - numhess, ord='fro') / linalgnorm(anhess, ord='fro')

# ## Display the relative error

# +
nbins = round(10*log(j_))
figure()

p = ones((1, len(err))) / len(err)

option = namedtuple('option', 'n_bins')
option.n_bins = nbins
[n, x] = HistogramFP(err.T, p, option)
b = bar(x[:-1], n[0], width=x[1]-x[0], facecolor= [.7, .7, .7])
plt.grid(True)
title('Relative error');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

