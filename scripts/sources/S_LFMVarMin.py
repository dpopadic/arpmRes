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

# # S_LFMVarMin [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_LFMVarMin&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-csvs-reg-load-copy-2).

# ## Prepare the environment

# +
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import trace, array, abs
from numpy.linalg import solve, pinv

import matplotlib.pyplot as plt

plt.style.use('seaborn')

# input parameters
n_ = 2  # dimension of target variable X
k_ = 1  # dimension of factors Z
mu_X = array([[-0.5], [1]])  # expectation of target variable X
sigma2_X = array([[1, .1], [.1, .2]])  # covariance of target variable X
beta = array([[1,1]]).T  # loadings
lam = 0.5  # exposure to the loadings

sigma2 = sigma2_X  # scale matrix equal covariance of target variable
# -

# ## Optimal loadings

beta_t_opt = solve(beta.T@solve(sigma2_X,beta),beta.T).dot(pinv(sigma2_X))

# ## Minimum variance portfolio

h_mv = (lam*beta_t_opt).T

# ## Distribution of factors Z = beta_t_opt@X

mu_Z = beta_t_opt@mu_X
sigma2_Z = beta_t_opt@sigma2_X@beta_t_opt.T

# ## r-square

R2 = trace((beta@beta_t_opt@sigma2_X).dot(pinv(sigma2))) / trace(sigma2_X.dot(pinv(sigma2)))
diff = abs((R2 - (k_ / n_)))
