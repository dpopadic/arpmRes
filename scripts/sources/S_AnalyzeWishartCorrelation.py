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

# # S_AnalyzeWishartCorrelation [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_AnalyzeWishartCorrelation&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-cor-norm-wish-marg).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import arange, array, zeros, sort, argsort, sqrt
from numpy.linalg import eig

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, ylim, subplots, ylabel, \
    xlabel

plt.style.use('seaborn')

from ARPM_utils import save_plot

# input parameters
sigvec = [1, 1]
nu = 15
# -

# ## Compute correlations and condition numbers as a function of rho

# +
corrrange = arange(-0.99, 1, 0.01)  # range of the parameter rho
n_ = len(corrrange)
cr_12 = zeros((n_, 1))
condnb = zeros((n_, 1))

for n in range(n_):
    rho = corrrange[n]
    cr_12[n] = sqrt(2) * rho / sqrt(1 + rho ** 2)  # correlation

    Cv_X = array([[1, cr_12[n]], [cr_12[n], 1]])  # covariance (=correlation) matrix

    Diag_lambda2, e = eig(Cv_X)
    lambda2 = Diag_lambda2
    lambda2, order = sort(lambda2)[::-1], argsort(lambda2)[::-1]

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
