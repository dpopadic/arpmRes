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

# # S_GenRegressionLFM [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_GenRegressionLFM&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-lfmgen-time).

# ## Prepare the environment

# +
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import reshape, array, ones, zeros, diag, eye, r_

from scipy.linalg import kron

import matplotlib.pyplot as plt

from quadprog import quadprog
plt.style.use('seaborn')
# -

# ## Computations

# +
# set inputs of target variable X and factor Z
m_X = array([[1], [0]])
m_Z = array([[-1]])
m_jointXZ = r_[m_X, m_Z]  # joint expectation

s2_X = array([[1, .5], [.5, 1]])
s_XZ = array([[.6], [.7]])
s2_Z = array([[1]])
s2_jointXZ = r_[r_['-1',s2_X, s_XZ], r_['-1',s_XZ.T, s2_Z]]  # joint covariance

n_ = m_X.shape[0]  # target dimension
k_ = m_Z.shape[0]  # number of factors
i_n = eye(n_)
i_k = eye(k_)

# set inputs for quadratic programming problem
d = np.diagflat(1 / diag(s2_X))
pos = d@s_XZ
g = -pos.flatten()
q = kron(s2_Z, d)

# set bound constraints
lb = 0.8*ones((n_*k_, 1))
ub = 1.2*ones((n_*k_, 1))

# compute optimal loadings
b = quadprog(q, g, None, None, lb, ub)

beta = reshape(b, (n_, k_),'F')
alpha = m_X - beta@m_Z

# joint distribution of residulas U and factor Z
m = r_[r_['-1',i_n, - beta], r_['-1',zeros((k_, n_)), i_k]]

m_jointUZ = m@m_jointXZ - r_[alpha, zeros((k_, 1))]  # joint expectation
s2_jointUZ = m@s2_jointXZ@m.T  # joint covariance
