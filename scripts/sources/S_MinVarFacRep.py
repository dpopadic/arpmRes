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

# # S_MinVarFacRep [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_MinVarFacRep&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-optim-pseudo-inv-lo).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import arange, array, ones, zeros
from numpy.linalg import solve
from numpy.random import rand

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, ylabel, \
    xlabel, title

plt.style.use('seaborn')

from ARPM_utils import save_plot

# input parameters
n_ = 100  # max market dimension
nstep = arange(5,n_+1)  # grid of market dimensions
s2_Z_ = array([[1]])  # variance of factor

stepsize = len(nstep)
s2_P_Z_MV = zeros((stepsize, 1))
s2_P_Z = zeros((stepsize, 1))

for n in range(stepsize):  # set covariance of the residuals
    d = rand(nstep[n], 1)
    s2_U = np.diagflat(d * d)

    # ## Compute the low-rank-diagonal covariance of the market
    beta = rand(nstep[n], 1)  # loadings
    s2_P = beta@s2_Z_@beta.T + s2_U

    # ## Compute the pseudo inverse of beta associated with the inverse covariance of the P&L's
    sig2_MV = np.diagflat(1 / (d * d))
    betap_MV = solve(beta.T@sig2_MV@beta,beta.T@sig2_MV)
    # NOTE: betap_MV does not change if we set sig2_MV = inv(s2_P)

    # ## Compute an arbitrary pseudo inverse of beta
    sig = rand(nstep[n], nstep[n])
    sig2 = sig@sig.T
    betap = solve(beta.T@sig2@beta,beta.T@sig2)

    # ## Compute the variances of the factor-replicating portfolio P&L
    s2_P_Z_MV[n] = betap_MV@s2_P@betap_MV.T
    s2_P_Z[n] = betap@s2_P@betap.T  # ## Plot the variances for each market dimension

figure()

plot(nstep, s2_P_Z_MV, 'b', linewidth=1.5, markersize=2)
plot(nstep, s2_P_Z, color= [.9, .3, 0], lw= 1.5, markersize=2)
plot(nstep, s2_Z_[0]*ones(stepsize), color= [.5, .5, .5], lw= 1.5, markersize=2)
plt.tight_layout()
xlabel(r'$\bar{n}$')
ylabel('variance')
title('Minimum variance factor-replicating portfolio')
h = legend(['$\sigma^2_{\Pi^{MV}_Z}$', '$\sigma^2_{\Pi_Z}$', '$\sigma^2_{Z}$']);
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
