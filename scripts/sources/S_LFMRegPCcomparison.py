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

# # S_LFMRegPCcomparison [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_LFMRegPCcomparison&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-end-expl-fact-lfm).

# ## Prepare the environment

# +
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import trace, array, zeros, eye, r_

import matplotlib.pyplot as plt

plt.style.use('seaborn')

from pcacov import pcacov

# input parameters
n_ = 2  # dimension of target variable X
k_ = 1  # dimension of factor Z
mu_Xemb = array([[1,0, - 1]]).T  # joint expectation of target X and factor Z
sigma2_Xemb = array([[1, .5, .6],
               [.5,  1, .7],
               [.6, .7,  1]])  # joint covariance of target X and factor Z
alpha_X_Reg = array([[1.6, .7]]).T  # optimal regression coefficients
beta_XZ_Reg = array([[.6, .7]]).T
# -

# ## Regression coefficients

alpha_Reg = r_[alpha_X_Reg, zeros((k_, 1))]
beta_Reg = r_[r_['-1',zeros((n_, n_)), beta_XZ_Reg],
              r_['-1',zeros((k_, n_)), eye(k_)]]

# ## Parameters of regression recovered embedding target

mu_Xtilde_Reg = alpha_Reg + beta_Reg@mu_Xemb
sigma2_Xtilde_Reg = beta_Reg@sigma2_Xemb@beta_Reg.T

# ## Principal-component coefficients

e, _ = pcacov(sigma2_Xemb)
beta_PC = e[:, :k_]@e[:, :k_].T
alpha_PC = mu_Xemb - beta_PC@mu_Xemb

# ## Parameters of principal-component recovered embedding target

mu_Xtilde_PC = alpha_PC + beta_PC@mu_Xemb
sigma2_Xtilde_PC = beta_PC@sigma2_Xemb@beta_PC.T

# ## r-squares

R2_Reg = 1 - (trace((beta_Reg - eye(n_ + k_))@sigma2_Xemb@(beta_Reg - eye(n_ + k_)).T) / trace(sigma2_Xemb))
R2_PC = 1 - (trace((beta_PC - eye(n_ + k_))@sigma2_Xemb@(beta_PC - eye(n_ + k_)).T) / trace(sigma2_Xemb))
