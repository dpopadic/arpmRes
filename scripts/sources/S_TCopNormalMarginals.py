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

# # S_TCopNormalMarginals [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_TCopNormalMarginals&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-2-ex-tcop-giv-marg).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import arange, prod, array, zeros, r_

from scipy.stats import norm, t

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, scatter, subplots, ylabel, \
    xlabel, title
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

plt.style.use('seaborn')
np.seterr(invalid='ignore')

from ARPM_utils import save_plot
from StudentTCopulaPdf import StudentTCopulaPdf
from Tscenarios import Tscenarios

# input parameters
j_ = 1000  # number of simulations
mu = array([[0], [0]])  # location
rho = 0.2  # correlation
sigvec = array([[1], [1]])  # standard deviations
nu = 10  # degrees of freedom

# grid in the unit square
Grid = arange(0, 1 + 0.05, 0.05)
nGrid = len(Grid)
# -

# ## Compute pdf of X by means of Sklar.T theorem

# +
c2 = array([[1, rho], [rho, 1]])  # correlation matrix
sigma2 = np.diagflat(sigvec)@c2@np.diagflat(sigvec)  # dispersion matrix

f_U = zeros((nGrid, nGrid))
f_X = zeros((nGrid, nGrid))
for n in range(nGrid):
    for m in range(nGrid):
        u = array([[Grid[n]], [Grid[m]]])
        f_U[n, m] = StudentTCopulaPdf(u, nu, mu, sigma2)  # pdf of copula
        f_X[n, m] = f_U[n, m]*prod(norm.pdf(norm.ppf(u, mu, sigvec), mu, sigvec))
# -

# ## Generate moment matching t-simulations

optionT = namedtuple('optionT', 'dim_red stoc_rep')
optionT.dim_red = 0
optionT.stoc_rep = 0
Z = Tscenarios(nu, mu, sigma2, j_, optionT, 'Riccati')

# ## Generate draws from the copula

U_1 = t.cdf((Z[0] - mu[0]) / sigvec[0], nu)  # grade 1
U_2 = t.cdf((Z[1] - mu[1]) / sigvec[1], nu)  # grade 2
U = r_[U_1, U_2]  # joint realizations from the required copula

# ## Generate draws of X from the grades and the inverse of normal marginals

X_1 = norm.ppf(U_1, mu[0], sigvec[0])
X_2 = norm.ppf(U_2, mu[1], sigvec[1])
X = r_[X_1[np.newaxis,...], X_2[np.newaxis,...]]  # joint realizations

# ## Display the pdf of X

# +
xx_1 = norm.ppf(Grid, mu[0], sigvec[0])
xx_2 = norm.ppf(Grid, mu[1], sigvec[1])
[x_1, x_2] = np.meshgrid(xx_1, xx_2)

f, ax = subplots(1, 1, subplot_kw=dict(projection='3d'))
ax.view_init(30,-120)
ax.plot_surface(x_1, x_2, f_X.T, cmap=cm.viridis,vmin=np.nanmin(f_X),vmax= np.nanmax(f_X))
xlabel('$X_1$',labelpad=10)
ylabel('$X_2$',labelpad=10)
title('Joint pdf of X');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
# -

# ## Scatter-plot of X_1 against X_2

figure()
scatter(X[0], X[1], s=10, c=[.5, .5, .5], marker='.')
xlabel('$X_1$')
ylabel('$X_2$')
title('Scatter-plot of X');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
