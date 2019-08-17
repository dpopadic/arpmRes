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

# # S_FPCopulaHistoricalComb [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_FPCopulaHistoricalComb&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-2-ex-cmacomb-hist).

# ## Prepare the environment

# +
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import array, ones, zeros, linspace

from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

plt.style.use('seaborn')

from CopMargComb import CopMargComb
from CopMargSep import CopMargSep

# historical marginal scenarios
Y = array([[-0.08, -0.04, -0.05, 0.09], [0.01, 0.05, -0.01, 0.03]])
n_, j_ = Y.shape
pj = ones((1, j_)) / j_  # flat Flexible probabilities

# joint scenarios of the grades
U = array([[0.96, 0.50, 0.80, 0.14, 0.42, 0.92], [0.79, 0.96, 0.66, 0.04, 0.85, 0.93]])
# -

# ## Deduce the ordered grid associated with the historical scenarios R

y_, u_,_ = CopMargSep(Y, pj)

# ## Compute the grid of significant evaluation nodes and cdf grid

# +
eta = 0.06
k_ = 6

y = zeros((n_, k_))
u = zeros((n_, k_))
for n in range(n_):
    interp = interp1d(u_[n,:], y_[n,:],fill_value='extrapolate')
    a = interp(eta)  # lower quantile
    b = interp(1 - eta)  # upper quantile
    y[n,:] = linspace(a, b, k_)
    interp = interp1d(y_[n,:], u_[n,:],fill_value='extrapolate')
    u[n,:] = interp(y[n,:])
# -

# ## Compute the joint scenarios through the CMA (combination) routine

X = CopMargComb(y, u, U)
