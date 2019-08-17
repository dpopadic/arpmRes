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

# # S_ToeplitzMatrix [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_ToeplitzMatrix&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=EigToepStruct).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import ones, sort, argsort, diagflat, eye
from numpy.linalg import eig

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot

plt.style.use('seaborn')

from ARPM_utils import save_plot

# Inputs
n_ = 200  # dimension of the matrix
rho = 0.9  # decay factor
# -

# ## Build Toeplitz matrix

t = eye(n_)
for n in range(n_ - 1):
    t = t + rho ** n * (diagflat(ones((n_ - n, 1)), n) + diagflat(ones((n_ - n, 1)), -n))

# ## Perform spectral decomposition

Diag_lambda2, e = eig(t)
lambda2, index = sort(Diag_lambda2)[::-1], argsort(Diag_lambda2)[::-1]
e = e[:, index]

# ## Plot first eigenvectors

figure()
color = [[0, 0.4470, 0.7410], [0.8500, 0.3250, 0.0980],[0.9290, 0.6940, 0.1250]]
for n in range(3):
    h = plot(e[:, n], color=color[n])
plt.grid(True);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
