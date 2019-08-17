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

# # S_AnalyzeNormalCorrelation [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_AnalyzeNormalCorrelation&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-cor-norm-2).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import arange, array, zeros, sort, argsort, diag, sqrt
from numpy.linalg import eig

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, ylim, subplots, ylabel, \
    xlabel

plt.style.use('seaborn')

from ARPM_utils import save_plot

# input parameters
mu = array([0, 0])
sigvec = array([1, 1])
# -

# ## Compute correlations and condition numbers as a function of rho

# +
corrrange = arange(-0.99, 1, 0.01)
n_ = len(corrrange)
cr_12 = zeros((n_, 1))
condnb = zeros((n_, 1))

for n in range(n_):
    rho = corrrange[n]
    sig2 = array([[sigvec[0] ** 2, rho * sigvec[0] * sigvec[1]], [rho * sigvec[0] * sigvec[1], sigvec[1] ** 2]])

    Cv_X = sig2  # covariance matrix
    Sd_X = sqrt(diag(Cv_X))  # standard deviation vector
    Cr_X = np.diagflat(1 / Sd_X)@Cv_X@np.diagflat(1 / Sd_X)  # correlation matrix

    Diag_lambda2, e = eig(Cv_X)
    lambda2 = Diag_lambda2
    lambda2, order = sort(lambda2)[::-1], argsort(lambda2)[::-1]

    cr_12[n] = Cr_X[0, 1]  # correlation
    condnb[n] = lambda2[1] / lambda2[0]  # condition number
# -

# ## Display correlations and condition numbers as a function of rho

# +
f, ax = subplots(2, 1)
plt.sca(ax[0])
plot(corrrange, cr_12)
ylim([-1, 1])
plt.grid(True)
xlabel(r'$\rho$')
ylabel('correlation')

plt.sca(ax[1])
plot(corrrange, condnb)
ylim([0, 1])
plt.grid(True)
xlabel(r'$\rho$')
ylabel('condition ratio')
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

