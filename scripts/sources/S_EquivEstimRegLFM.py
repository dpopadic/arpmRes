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

# # S_EquivEstimRegLFM [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_EquivEstimRegLFM&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=EquivFormMLFPEstimReg).

# ## Prepare the environment
import os
# # +
import os.path as path
import sys

from scipy.io import loadmat

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import array
import numpy as np
from numpy import ones, diag, r_, diagflat
from numpy import sum as npsum
from numpy.linalg import pinv, norm
from numpy.random import rand, seed
from numpy.random import multivariate_normal as mvnrnd

import matplotlib.pyplot as plt

plt.style.use('seaborn')

from OrdLeastSquareFPNReg import OrdLeastSquareFPNReg

# input parameters
n_ = 6  # target dimension
k_ = 3  # number of factors
t_ = 1000  # time series len
p = ones((1, t_)) / t_  # Flexible Probabilities
w = rand(1, t_)  # weights

mu = 5 * ones(n_ + k_)

c = rand(n_ + k_, n_ + k_)
sig2 = c@c.T


XZ = mvnrnd(mu, sig2, size=(t_))
XZ = XZ.T  # observations of target variables and factors
# -

# ## Compute MLSFP estimators, for given weights w

X = XZ[:n_, :]
Z = XZ[n_:n_ + k_, :]
pw = p * w
alpha, beta, *_ = OrdLeastSquareFPNReg(X, Z, pw / npsum(pw))

# ## Compute alternative compact formulation

Z_ = r_[ones((1, t_)), Z]
XZ_ = r_[X, Z_]
s2_XZ_ = XZ_@diagflat(pw)@XZ_.T
s_XZ_ = s2_XZ_[:n_, n_:n_ + k_+1]
s2_Z_ = s2_XZ_[n_:n_ + k_+1, n_:n_ + k_+1]
b = s_XZ_.dot(pinv(s2_Z_))

# ## Compare the expressions

err = norm(r_['-1', alpha, beta] - b, ord='fro')
