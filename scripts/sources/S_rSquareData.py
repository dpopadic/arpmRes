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

# # S_rSquareData [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_rSquareData&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-ex-unv-rsquare).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import ones, round, log, sqrt
from numpy import sum as npsum

import numpy as np
np.seterr(divide='ignore')

from scipy.io import loadmat

import matplotlib.pyplot as plt

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from MultivRsquare import MultivRsquare

# input parameters
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_ExSummary'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_ExSummary'), squeeze_me=True)

t_ = db['t_']
epsi = db['epsi']
v = db['v']

j_ = t_  # dimension of data set
p = ones((1, j_)) / j_  # uniform Flexible Probabilities
x = epsi[0]  # model data
x_tilde = epsi[1]  # fit data
# log values
y = round(log(v[0, 1:]))  # model data
y_tilde = round(log(v[1, 1:]))  # fit data

z = x_tilde
# -

# ## Compute the residuals

u_x = x - x_tilde
u_y = y - y_tilde

# ## Compute the data mean and variance

# +
m_x = npsum(p * x)
m_y = npsum(p * y)
m_z = npsum(p * z)
m_u_x = npsum(p * u_x)
m_u_y = npsum(p * u_y)

sigma2_x = npsum(p * (x - m_x) ** 2,keepdims=True)
sigma2_y = npsum(p * (y - m_y) ** 2,keepdims=True)
sigma2_u_x = npsum(p * (u_x - m_u_x) ** 2,keepdims=True)
sigma2_u_y = npsum(p * (u_y - m_u_y) ** 2,keepdims=True)
# -

# ## Compute the r-squared

r2_x = MultivRsquare(sigma2_u_x, sigma2_x, 1 / sigma2_x)
r2_y = MultivRsquare(sigma2_u_y, sigma2_y, 1 / sigma2_y)

# ## Compute the correlation

rho_HFP = npsum(p * (x - m_x) * (z - m_z)) / (sqrt(npsum(p * (x - m_x) ** 2))*sqrt(npsum(p * (z - m_z) ** 2)))
