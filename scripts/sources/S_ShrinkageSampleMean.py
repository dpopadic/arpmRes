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

# # S_ShrinkageSampleMean [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_ShrinkageSampleMean&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ShrinkageEstLocMatlab).

# ## Prepare the environment

# +
import os.path as path
import sys

from numpy import minimum
from numpy import maximum

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import zeros, cov, mean
from numpy import max as npmax, sum as npsum
from numpy.linalg import eig
from numpy.random import rand
from numpy.random import multivariate_normal as mvnrnd

import matplotlib.pyplot as plt

plt.style.use('seaborn')

# initialize variables

i_ = 10
t_ = 30
mu = rand(i_, 1)
l = rand(i_, i_) - 0.5
sigma2 = l@l.T
# -

# ## Generate normal sample

Epsi = mvnrnd(mu.flatten(), sigma2, t_).T

# ## Estimate sample parameters

mu_hat = mean(Epsi, 1,keepdims=True)
sigma2_hat = cov(Epsi)

# ## Perform shrinkage of location parameter

# +
# target
b = zeros((i_, 1))

# compute optimal weight
lambda_hat, _ = eig(sigma2_hat)  # eigenvalues
a = (2 / t_)*(npsum(lambda_hat) - 2*npmax(lambda_hat))
c = a / ((mu_hat - b).T@(mu_hat - b))
# restrict to sensible weight
c = maximum(0, minimum(c, 1))

# shrink
mu_bar = (1 - c)*mu_hat + c*b
# -

# ## Show results

print('shrinkage confidence: %f' %c)
print('sample mean estimator: ', mu_hat)
print('shrinkage estimator: ', mu_bar)
