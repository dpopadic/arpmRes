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

# # S_LFMRegCSLoadComparison [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_LFMRegCSLoadComparison&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-csvs-reg-load).

# ## Prepare the environment

# +
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import array, diag
from numpy.linalg import solve, pinv, norm

import matplotlib.pyplot as plt

plt.style.use('seaborn')

# input parameters
n_ = 2  # dimension of target variable X
k_ = 1  # dimension of factors Z
mu_X = array([[-0.5], [1]])  # expectation of target variable X
sigma2_X = array([[1, .1], [.1, .2]])  # covariance of target variable X
beta = array([[1],[1]])  # loadings
# -

# ## Compute the extraction matrix using scale matrix sigma2 = Diag(V{X}

sigma2 = np.diagflat(diag(sigma2_X))
psinv = solve((beta.T.dot(pinv(sigma2)))@beta, beta.T.dot(pinv(sigma2)))

# ## Compute the regression loadings and check if regression and cross-sectional loadings are different

beta_Reg_1 = (sigma2_X@psinv.T).dot(pinv(psinv@sigma2_X@psinv.T))
diff_1 = norm(beta_Reg_1 - beta)

# ## Compute the extraction matrix using scale matrix sigma2=sigma2_X

beta_t_2 = solve((beta.T.dot(pinv(sigma2_X)))@beta,beta.T.dot(pinv(sigma2_X)))

# ## Compute the regression loadings and check if regression and cross-sectional loadings are different

beta_Reg_2 = (sigma2_X@beta_t_2.T).dot(pinv(beta_t_2@sigma2_X@beta_t_2.T))
diff_2 = norm(beta_Reg_2 - beta)
