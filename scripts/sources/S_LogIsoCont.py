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

# # S_LogIsoCont [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_LogIsoCont&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExLogNIsoContour).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import arange, array, zeros, pi, eye, log, exp, sqrt, \
    r_
from numpy import min as npmin, max as npmax
from numpy.linalg import solve, det
from numpy.random import multivariate_normal as mvnrnd

import matplotlib.pyplot as plt
from matplotlib.pyplot import scatter, ylabel, \
    xlabel

plt.style.use('seaborn')

from ARPM_utils import save_plot

# parameters
mu = array([[0.8], [0.8]])
sigma2 = array([[1.2, 0], [0, 1]])
j_ = 40000  # number of simulations
# -

# ## Generate the bivariate lognormal simulations

X = mvnrnd(mu.flatten(), sigma2, j_)
Y = exp(X)

# ## Select an equispaced grid and compute the lognormal pdf

# +
x1 = arange(0.01,7,0.1)
x2 = arange(0.01,7,0.1)
X1, X2 = np.meshgrid(x1, x2)
lX1 = log(X1)
lX2 = log(X2)
z = r_[lX2.flatten()[np.newaxis,...], lX1.flatten()[np.newaxis,...]]
s = len(x1)*len(x2)
f = zeros(s)
for i in range(s):
    f[i] = exp(-1 /2 *((z[:,[i]]-mu).T)@solve(sigma2,eye(sigma2.shape[0]))@(z[:, [i]]-mu))/(2*pi*sqrt(det(sigma2))*(X1.flatten()[i]*X2.flatten()[i]))

f = np.reshape(f, (len(x2), len(x1)), order='F')
# -

# ## Display the iso-contours and the scatter plot of the corresponding sample

plt.contour(X1, X2, f, arange(0.01, 0.03,0.005), colors='b',lw=1.5)
scatter(Y[:, 0], Y[:, 1], 1, [.3, .3, .3], '.')
plt.axis([npmin(x1), npmax(x1),npmin(x2), npmax(x2)])
xlabel(r'$x_1$')
ylabel(r'$x_2$');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
