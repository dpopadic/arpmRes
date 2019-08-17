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

# # S_LognormPrincipalBasis [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_LognormPrincipalBasis&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=EBRandGeomLogN2).

# ## Prepare the environment

# +
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import array, ones, diag, exp

import matplotlib.pyplot as plt

plt.style.use('seaborn')

from pcacov import pcacov
# -

# ## input parameters

m = [0.17, 0.06]  # (normal) expectation
svec = [0.24, 0.14]  # (normal) standard deviation
rho = 0.15  # (normal) correlation

# ## Compute lognormal expectation and covariance

c2_ = array([[1, rho], [rho, 1]])  # (normal) correlation matrix
s2 = np.diagflat(svec)@c2_@np.diagflat(svec)  # (normal) covariance matrix
mu = exp(m + 0.5*diag(s2))  # expectation
sig2 = np.diagflat(mu)@(exp(s2) - ones((2, 1)))@np.diagflat(mu)  # covariance matrix

# ## Principal Component Analysis of the covariance matrix

[e, lambda2] = pcacov(sig2)

# ## Verify that the principal basis is orthogonal both in the euclidean and in the statistical sense

euclidean_inner_prod = e.T@e
statistical_inner_prod = e.T@sig2@e
