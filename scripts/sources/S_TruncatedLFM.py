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

# # S_TruncatedLFM [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_TruncatedLFM&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-lfmtrunc).

# ## Prepare the environment

# +
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import array, zeros, diag, eye, r_
from numpy.linalg import pinv

import matplotlib.pyplot as plt

plt.style.use('seaborn')

# input parameters
n_ = 2  # target dimension
k_ = 1  # number of factors
sig2_XZ = array([[4, 9.5, -1.8], [9.5, 25, -4.5], [-1.8, -4.5, 1]])  # joint covariance of target variables and factors
# -

# ## Compute optimal loadings

# +
sig_XZ = sig2_XZ[:n_, n_:n_ + k_]
sig2_Z = sig2_XZ[n_:n_ + k_, n_:n_ + k_]

b = sig_XZ.dot(pinv(sig2_Z))
# -

# ## Compute truncated joint covariance of residuals and factors

m = r_[r_['-1', eye(n_), -b], r_['-1', zeros((k_, n_)), eye(k_)]]
sig2_UZ = m@sig2_XZ@m.T
sig2_UZtrunc = r_[r_['-1', np.diagflat(diag(sig2_UZ[:n_, :n_])), zeros((n_, k_))], r_[
    '-1', zeros((k_, n_)), sig2_UZ[n_:n_ + k_, n_:n_ + k_]]]

# ## Compute truncated covariance of target variables

m_tilde = r_['-1', eye(n_), b]
sig2_Xtrunc = m_tilde@sig2_UZtrunc@m_tilde.T
