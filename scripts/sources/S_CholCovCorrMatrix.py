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

# # S_CholCovCorrMatrix [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_CholCovCorrMatrix&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-chol-cov-corr-matrix-copy-1).

# ## Prepare the environment

# +
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy.linalg import norm as linalgnorm, cholesky
from numpy.random import randn

import matplotlib.pyplot as plt

plt.style.use('seaborn')

from cov2corr import cov2corr
# -

# ## Choose an arbitrary n_ x n_ positive definite covariance matrix sigma2

# initial setup
n_ = 50
a = randn(n_, n_)
sigma2 = a@a.T

# ## Compute the n_ x n_ positive definite correlation matrix c2

sigma_vec, c2 = cov2corr(sigma2)

# ## Perform the Cholesky decomposition of c2

l_tilde_c = cholesky(c2)

# ## Compute the symmetric and lower triangular matrix l_tilde_s

l_tilde_s = np.diagflat(sigma_vec)@l_tilde_c

# ## Check that sigma2 = l_tilde_s@l_tilde_s.T is the Cholesky decomposition of sigma2

sigma2_chol = l_tilde_s@l_tilde_s.T
check1 = linalgnorm(sigma2_chol - sigma2, ord='fro')
check2 = linalgnorm(l_tilde_s - cholesky(sigma2), ord='fro')
