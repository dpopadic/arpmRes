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

# # S_DisplayStudentTCopulaPdf [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_DisplayStudentTCopulaPdf&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-2-ex-tcop-trad).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import arange, array, zeros, r_

from scipy.stats import t

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, scatter, subplots, ylabel, \
    xlabel, title
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('seaborn')

from ARPM_utils import save_plot
from Tscenarios import Tscenarios
from StudentTCopulaPdf import StudentTCopulaPdf

# input parameters
j_ = 3000  # number of simulations
mu = array([[0],[0]])  # location
rho = 0  # correlation
sigvec = array([[1], [2]])  # standard deviations
nu = 4  # degrees of freedom

# grid in the unit square
Grid = arange(0,1.05,0.05)
nGrid = len(Grid)
# -

# ## Compute the pdf of the copula

# +
f_U = zeros((nGrid, nGrid))
c2 = array([[1, rho], [rho, 1]])  # correlation matrix
sigma2 = np.diagflat(sigvec)@c2@np.diagflat(sigvec)  # dispersion matrix

for n in range(nGrid):
    for m in range(nGrid):
        u = r_[Grid[n], Grid[m]].reshape(-1,1)
        f_U[n,m] = StudentTCopulaPdf(u, nu, mu, sigma2)  # ## Generate moment matching t-simulations

optionT = namedtuple('option', 'dim_red stoc_rep')
optionT.dim_red = 0
optionT.stoc_rep = 0
X = Tscenarios(nu, mu, sigma2, j_, optionT, 'PCA')
# -

# ## Generate draws from the copula of the t distribution

U_1 = t.cdf((X[[0]] - mu[0]) / sigvec[0], nu)  # grade 1
U_2 = t.cdf((X[[1]] - mu[1]) / sigvec[1], nu)  # grade 2
U = r_[U_1, U_2]  # joint realizations from the required copula

# ## Display the pdf of the t-copula

# +
u_1, u_2 = np.meshgrid(Grid, Grid)

f,ax = subplots(1,1,subplot_kw={'projection':'3d'})
ax.plot_surface(u_1, u_2, f_U.T)
xlabel('Grade $U_1$')
ylabel('Grade $U_2$')
str = 'Pdf of t - Copula with correlation =  % .2f'%rho
title(str);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
# -

# ## Scatter-plot of the t-copula scenarios

figure()
scatter(U[0], U[1], s=10, c=[.5, .5, .5], marker='*')
xlabel('Grade $U_1$')
ylabel('Grade $U_2$')
title('Grade scenarios');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
