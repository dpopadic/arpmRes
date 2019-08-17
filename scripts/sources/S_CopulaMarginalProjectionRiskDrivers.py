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

# # S_CopulaMarginalProjectionRiskDrivers [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_CopulaMarginalProjectionRiskDrivers&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-cmprojection-copy-2).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import arange, ones, zeros, cumsum, squeeze, \
    abs, sqrt, tile, r_

from scipy.linalg import expm
from scipy.io import loadmat
from scipy.stats import t as tstu

import matplotlib.pyplot as plt

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from Tscenarios import Tscenarios
from CopMargComb import CopMargComb

# -

# ## Upload the database db_CopulaMarginalRiskDrivers

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_CopulaMarginalRiskDrivers'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_CopulaMarginalRiskDrivers'),
                 squeeze_me=True)  # output db from S_CopulaMarginalEstimationRiskDrivers

d_ = db['d_']
delta_t = db['delta_t']
x_MVOU = db['x_MVOU']
x_HST = db['x_HST']
x_VG = db['x_VG']
marginals_grid = db['marginals_grid']
marginals_cdfs = db['marginals_cdfs']
mu_epsi = db['mu_epsi'].reshape(-1, 1)
c2_hat = db['c2_hat']
nu = db['nu']
mu_x_HST = db['mu_x_HST']
mu_MVOU = db['mu_MVOU']
eta = db['eta']
kappa = db['kappa']
y = db['y']
y_bar = db['y_bar']
theta_MVOU = db['theta_MVOU']
# -

# ## Fix an investment horizon tau multiple of the estimation step delta_t

# +
# Initialize projection variables
horiz = 30  # horizon =30 days
u = arange(0, delta_t + horiz, delta_t)
t_sim = len(u) - 1
j_ = 4000

# initialize arrays
dY = zeros((1, j_, t_sim))
Y = zeros((1, j_, t_sim + 1))
dX_HST = zeros((1, j_, t_sim))
X_MVOU = zeros((d_, j_, t_sim))
x_0_MVOU = tile(x_MVOU[:, [-1]], (1, j_))
dT = zeros((1, j_, t_sim))
dX_VG = zeros((1, j_, t_sim))

# initialize variance
Y[0, :, 0] = y[-1] * ones(j_)

# create paths
for t in range(t_sim):
    # ## Generate scenarios for the invariants

    # simulate scenarios for the grades U by using the estimated correlation matrix c2
    optionT = namedtuple('option', 'dim_red stoc_rep')
    optionT.dim_red = 0
    optionT.stoc_rep = 0
    U = tstu.cdf(Tscenarios(nu, mu_epsi, c2_hat, j_, optionT, 'PCA'), nu)

    # Retrieve the estimated marginals cdf's from S_CopulaMarginalEstimationRiskDrivers and combine them with the
    # scenarios for the grades U, to generate joint scenarios for the invariants
    Epsi = CopMargComb(marginals_grid, marginals_cdfs, U)

    # ## Apply the incremental step routine to generate Monte Carlo paths for the risk drivers
    # project the Heston process for the log-values
    dY[0, :, t] = -kappa * (Y[0, :, t] - y_bar) * delta_t + eta * sqrt(Y[0, :, t]) * Epsi[1]
    Y[0, :, t + 1] = abs(Y[0, :, t] + dY[0, :, t])
    dX_HST[0, :, t] = mu_x_HST * delta_t + sqrt(Y[0, :, t]) * Epsi[0]

    # project the MVOU process for the shadow short-rates
    if t_sim > 1 and t > 1:
        x_0_MVOU = X_MVOU[:, :, t - 1]

    X_MVOU[:, :, t] = expm(-theta_MVOU * delta_t) @ x_0_MVOU + tile(mu_MVOU[..., np.newaxis] * delta_t, (1, j_)) + Epsi[
                                                                                                                   2: 4,
                                                                                                                   :]  # shadow rates

    # VG increments
    dX_VG[:, :, t] = Epsi[4, :]

X_HST = x_HST[-1] + r_['-1', zeros((j_, 1)), cumsum(dX_HST.squeeze(), 1)]  # log-stock
X_VG = x_VG[-1] + cumsum(dX_VG, 2)  # option strategy cumulative P&L (random walk)

print('The projected paths of the log - stock is stored in X_HST \nThe projected paths of the 2 - year and 7 - year '
      'shadow rates are stored in X_MVOU\nThe projected paths of the option strategy cumulative P & L '
      'are stored in X_VG')
