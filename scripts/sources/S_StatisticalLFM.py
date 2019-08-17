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

# # S_StatisticalLFM [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_StatisticalLFM&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-lfmhid-cor).

# ## Prepare the environment

# +
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import array, zeros, sort, argsort, diag, eye, sqrt, r_
from numpy.linalg import eig

import matplotlib.pyplot as plt

plt.style.use('seaborn')
# -

# ## Computations

# +
# set inputs
m_X = array([[1], [-1]])
s2_X = array([[4.1, 1.2], [1.2, 3.4]])

n_ = m_X.shape[0]  # target dimension
k_ = 1  # number of factors
i_n = eye(n_)

# compute correlation's spectral decomposition
s_X = sqrt(diag(s2_X))
c2 = np.diagflat(1 / s_X)@s2_X@np.diagflat(1 / s_X)  # correlation matrix

lambda2, e = eig(c2)

lambda2, order = sort(lambda2)[::-1], argsort(lambda2)[::-1]  # sort eigenvalues
e = e[:, order]  # sort eigenvectors

# compute optimal coefficients
e_k = e[:, :k_]
beta = np.diagflat(s_X)@e_k
gamma = e_k.T@np.diagflat(1 / s_X)
alpha = (i_n - beta@gamma)@m_X

# compute the parameters of the factor distribution
m_Z = gamma@m_X
s2_Z = gamma@s2_X@gamma.T

# joint distribution of residulas U and factor Z
m = r_[i_n - beta@gamma, gamma]

m_jointUZ = m@m_X - r_[alpha, zeros((k_, 1))]  # joint expectation
s2_jointUZ = m@s2_X@m.T  # joint covariance
