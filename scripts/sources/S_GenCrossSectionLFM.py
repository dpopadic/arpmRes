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

# # S_GenCrossSectionLFM [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_GenCrossSectionLFM&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-lfmgen-cross).

# ## Prepare the environment

# +
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import reshape, array, zeros, diag, eye, r_

from scipy.linalg import kron

import matplotlib.pyplot as plt

from quadprog import quadprog

plt.style.use('seaborn')
# -

# ## Computations

# +
# set  inputs
m_X = array([[-0.5], [1]])  # expectation of target variable X
s2_X = array([[1, .1], [.1, .2]])  # covariance of target variable X

beta = array([[1], [1]])  # loadings

n_ = m_X.shape[0]  # target dimension
k_ = beta.shape[1]  # number of factors
i_n = eye(n_)
i_k = eye(k_)

km = zeros((k_*n_, k_*n_))  # commutation matrix
for n  in range(n_):
    for k in range(k_):
        km = km + kron(i_k[:,[k]]@i_n[:, [n]].T, i_n[:,[n]]@i_k[:, [k]].T)

# set inputs for quadratic programming problem
invsigma2 = np.diagflat(1 / diag(s2_X))
pos = beta.T@invsigma2@s2_X
g = -pos.flatten()
q = kron(s2_X, beta.T@invsigma2@beta)
q_, _ = q.shape

# linear constraints
v = array([[-1, 1]])
d_eq = kron(i_k, v@s2_X)@km
b_eq = zeros((k_ ** 2, 1))

# compute extraction matrix
# options = optimoptions(('quadprog','MaxIter.T, 2000, .TAlgorithm','interior-point-convex'))
c = quadprog(q, g, d_eq, b_eq)

gamma = reshape(c, (k_, n_),'F')
alpha = (i_n - beta@gamma)@m_X

# joint distribution of residulas U and factor Z
m = r_[i_n - beta@gamma, gamma]

m_jointUZ = m@m_X - r_[alpha, zeros((k_, 1))]
s2_jointUZ = m@s2_X@m.T
