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

# # S_CopOpPoolUnifView [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_CopOpPoolUnifView&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=cop-norm-market).

# ## Prepare the environment

# +
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import array, ones

from scipy.stats import uniform

import matplotlib.pyplot as plt

plt.style.use('seaborn')

from CopulaOpinionPooling import CopulaOpinionPooling

# input parameters
X_pri = array([[9.27, - 15.94, 9.88, 22.13],
         [12.03, 3.59, 15.44, 9.78],
         [12.81, - 12.56, 20.58, 2.96]])  # scenarios of prior market variables
n_, j_ = X_pri.shape  # dimension of the market and number of scenarios
p = ones((1, j_)) / j_  # Flexible Probabilities

v = array([[1,-1,0],
     [0, 1, -1]])  # pick matrix
k_ = v.shape[0]  # number of views

c_full = ones((k_, 1)) - 1e-6  # full confidence levels
c = ones((k_, 1))*0.5  # half confidence levels

# View cdf's
# parameters of the uninformative views
a = [0, 0]
b = [0.02, 0.001]

# view cdf's
FZ_pos = [lambda x: uniform.cdf(x, a[0], b[0]), lambda x: uniform.cdf(x, a[1], b[1])]
# -

# ## Compute posterior market distribution with full confidence

X_updated, Z_pri, U, Z_pos, v_tilde, Z_tilde_pri, Z_tilde_pos = CopulaOpinionPooling(X_pri, p, v, c_full, FZ_pos)

# ## Compute posterior market distribution with confidence c=0.5

X_updated_c, _, _, _, _, _, _ = CopulaOpinionPooling(X_pri, p, v, c, FZ_pos)
