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

# # S_ShrinkageSampleCovariance [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_ShrinkageSampleCovariance&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerShrinkEstScatter).

# ## Prepare the environment

# +
import os.path as path
import sys

from numpy import minimum, maximum
from scipy.io import loadmat

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import trace, cov, eye
from numpy.random import rand
from numpy.random import multivariate_normal as mvnrnd

from scipy.stats import t

import matplotlib.pyplot as plt

plt.style.use('seaborn')

# initialize variables

i_ = 5
t_ = 30
mu = rand(i_, 1)
l = rand(i_, i_) - 0.5
sigma2 = l@l.T
# -

# ## Generate normal sample

Epsi = mvnrnd(mu.flatten(), sigma2, t_).T

# ## Estimate sample covariance

# mu_hat = mean(Epsi,2)
sigma2_hat = cov(Epsi, ddof=0)

# ## Perform shrinkage of dispersion parameter

# +
# target
sigma_target = trace(sigma2_hat)/i_*eye(i_)

# compute optimal weight
num = 0
for t in range(t_):
    num += trace((Epsi[:,[t]]@(Epsi[:, [t]].T) - sigma2_hat)@(Epsi[:,[t]]@(Epsi[:, [t]].T) - sigma2_hat)) / t_

den = trace((sigma2_hat - sigma_target)@(sigma2_hat - sigma_target))
gamma = num / (t_*den)
# restrict to sensible weight
gamma = maximum(0, minimum(gamma, 1))

# shrink
sigma2_c = gamma*sigma_target + (1 - gamma)*sigma2_hat
# -

# ## Show results

print('shrinkage confidence: ',gamma)
print('sample covariance estimator: ',sigma2_hat)
print('shrinkage estimator: ',sigma2_c)
