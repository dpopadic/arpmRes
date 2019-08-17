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

# # S_CopulaTestNormal [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_CopulaTestNormal&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerIIDCopTest).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

from tqdm import trange

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import ones, zeros, abs, r_

from scipy.stats import norm

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
plt.style.use('seaborn')

from ARPM_utils import save_plot
from InvarianceTestCopula import InvarianceTestCopula
from SWDepMeasure import SWDepMeasure

# input parameters
t_ = 100  # time series length
mu = 0  # expectation
sigma = 0.25  # standard deviation
lag_ = 10  # number of lags to focus on
# -

# ## Generate normal simulations and compute their absolute values

Epsi = norm.rvs(mu, sigma, (1, t_))
abs_Epsi = abs(Epsi)

# ## Estimate the SW measures of dependence

dep_absEpsi = zeros((lag_, 1))
for l in trange(lag_):
    probs = ones((1, t_ -(l+1))) / (t_ - (l+1))  # set flat Flexible probabilities
    dep_absEpsi[l] = SWDepMeasure(r_[abs_Epsi[[0],l+1:], abs_Epsi[[0],:-(l+1)]], probs)

# ## Plot copula pdf and measures of dependence for invariance test

f = figure(figsize=(12,6))
InvarianceTestCopula(abs_Epsi, dep_absEpsi, lag_);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

