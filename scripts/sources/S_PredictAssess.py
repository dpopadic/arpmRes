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

# # S_PredictAssess [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_PredictAssess&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-predictor-assess).

# ## Prepare the environment

# +
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import arange, ones, zeros, mean, sqrt
from numpy.random import randint, permutation

from scipy.stats import norm

import matplotlib.pyplot as plt

plt.style.use('seaborn')

from OrdLeastSquareFPNReg import OrdLeastSquareFPNReg
from RelEntropyMultivariateNormal import RelEntropyMultivariateNormal

# input parameters
t_ = 1000  # len of time series
beta = 1  # real value of beta
alpha = 0  # real value of alpha
sigma2 = 4  # real value of sigma
lsub = 200  # len of sub-samples
# -

# ## Generate simulations of factor

Z = norm.rvs(0, 1, [1, t_])

# ## Generate time series of residuals

U = norm.rvs(0, sqrt(sigma2), [1, t_])

# ## Compute simulations of target variable and time series

I = U + alpha + beta*Z

# ## Reshuffle the training set

# +
perm = permutation(arange(t_))
I_perm = I[0,perm].reshape(1,-1)
Z_perm = Z[0,perm].reshape(1,-1)

# number of samples
k_ = int(t_ / lsub)

MLobj = zeros((1, k_))
NonSobj = zeros((1, k_))
t_vec = range(t_)
for m in range(k_):
    t_in = arange(m*lsub, (m + 1)*lsub)  # in-sample observations
    t_out = np.setdiff1d(t_vec, t_in)  # out-of-sample observations
    # extract sub-samples
    I_in = I_perm[0,t_in].reshape(1,-1)
    I_out = I_perm[0,t_out].reshape(1,-1)

    Z_in = Z_perm[0,t_in].reshape(1,-1)
    Z_out = Z_perm[0,t_out].reshape(1,-1)

    # set flat flexible probabilities
    sub_t = I_in.shape[1]
    p = ones((1, sub_t)) / sub_t

    csub_t = I_out.shape[1]
    c_p = ones((1, csub_t)) / csub_t

    # maximum likelihood predictor
    alpha_OLSFP, beta_OLSFP, s2_OLSFP,_ = OrdLeastSquareFPNReg(I_in, Z_in, p)
    c_alpha_OLSFP, c_beta_OLSFP, c_s2_OLSFP,_= OrdLeastSquareFPNReg(I_out, Z_out, c_p)

    mu = alpha_OLSFP + beta_OLSFP*Z[0,-1]
    c_mu = c_alpha_OLSFP + c_beta_OLSFP*Z[0,-1]

    MLobj[0,m] = RelEntropyMultivariateNormal(mu, s2_OLSFP, c_mu, c_s2_OLSFP)

    # nonsensical predictor
    alpha_cap = 0
    beta_cap = I_in[0,-1]*Z_in[0,0]
    sigma2_cap = I_in[0,-1]**2*I_in[0,0] ** 2

    c_alpha_cap = 0
    c_beta_cap = I_out[0,-1]*Z_out[0,0]
    c_sigma2_cap = I_out[0,-1] ** 2*I_out[0,0] ** 2

    mu = alpha_cap + beta_cap*Z[0,-1]
    c_mu = c_alpha_cap + c_beta_cap*Z[0,-1]

    NonSobj[0,m] = RelEntropyMultivariateNormal(np.atleast_1d(mu), np.atleast_2d(sigma2_cap), np.atleast_1d(c_mu),
                                                np.atleast_2d(c_sigma2_cap))

vML = mean(MLobj)
vNonS = mean(NonSobj)
