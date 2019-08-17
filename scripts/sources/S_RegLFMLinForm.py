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

# # S_RegLFMLinForm [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_RegLFMLinForm&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-sym-reg-copy-3).

# ## Prepare the environment

# +
import os.path as path
import sys, os

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import array, zeros, r_
from numpy.linalg import norm as linalgnorm, pinv
from numpy.random import rand

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, subplots, hist

from ARPM_utils import save_plot

plt.style.use('seaborn')

# settings
n_ = 50  # number of market variables
k_ = 10  # number of observable factors

dist = zeros(100)

for i in range(100):
    # ## Generate j_ = 100 arbitrary parameters
    m_x = rand(n_, 1)
    m_z = rand(k_, 1)
    a = rand(n_+k_, n_+k_)
    s2_xz = a@a.T

    # ## Compute the coefficients of the classical formulation and the linear formulation

    # Classical formulation
    beta = s2_xz[:n_, n_:].dot(pinv(s2_xz[n_:, n_:]))
    alpha = m_x - beta@m_z

    # Linear formulation parameters
    e_xz_tilde = r_['-1',m_x, s2_xz[:n_, n_:] + m_x@m_z.T]
    e_z2_tilde = r_[r_['-1',array([[1]]), m_z.T],r_['-1',m_z, s2_xz[n_:, n_:] + m_z@m_z.T]]
    beta_tilde = e_xz_tilde.dot(pinv(e_z2_tilde))

    # Frobenius distance
    dist[i] = linalgnorm(r_['-1',alpha, beta] - beta_tilde, ord='fro')
# -

# ## Plot the histogram of the Frobenius norms

# +
f, ax = subplots(1,1)

hist(dist);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
