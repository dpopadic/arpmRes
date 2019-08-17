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

# # S_FullyExogenousLFM [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_FullyExogenousLFM&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-lfmsys-id-copy-1).

# ## Prepare the environment

# +
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import array, zeros, eye, r_

import matplotlib.pyplot as plt

plt.style.use('seaborn')

# set inputs
mu_X = array([[1],[-1]])
mu_Z = array([[1]])
mu_jointXZ = r_[mu_X, mu_Z]

sigma2_X = array([[4, 3], [3, 5]])
sigma_XZ = array([[3], [3]])
Sigma_Z = array([[3]])
sigma2_jointXZ = r_[r_['-1',sigma2_X, sigma_XZ], r_['-1',sigma_XZ.T, Sigma_Z]]

beta = array([[2], [1]])
alpha = array([[1], [-2]])

n_ = mu_X.shape[0]  # target dimension
k_ = mu_Z.shape[0]  # number of factors
i_n = eye(n_)
i_k = eye(k_)
# -

# ## Compute expectation and covariance of the joint distribution of U and Z

# +
m = r_[r_['-1',i_n, -beta], r_['-1',zeros((k_, n_)), i_k]]

mu_jointUZ = m@mu_jointXZ - r_[alpha, zeros((k_, 1))]
sigma2_jointUZ = m@sigma2_jointXZ@m.T
