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

# # S_FitSVI [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_FitSVI&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerImplVolSVI).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import arange, zeros, tile, r_

from scipy.io import loadmat, savemat

import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots, ylabel, \
    xlabel, title
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import struct_to_dict, save_plot
from SigmaSVI import SigmaSVI
from FitSigmaSVI import FitSigmaSVI

# parameters
y = 0  # risk free rate
# -

# ## Upload the data from db_ImpliedVol_SPX

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_ImpliedVol_SPX'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_ImpliedVol_SPX'), squeeze_me=True)  # implied volatility surface for SP500

db_ImpliedVol_SPX = struct_to_dict(db['db_ImpliedVol_SPX'])

dates = db_ImpliedVol_SPX.Dates
tau = db_ImpliedVol_SPX.TimeToMaturity
delta = db_ImpliedVol_SPX.Delta  # delta-moneyness
sigma_delta = db_ImpliedVol_SPX.Sigma

n_, k_, t_ = sigma_delta.shape
# -

# ## For each observation, use function FitSigmaSVI to compute the SVI parameters and function SigmaSVI to compute the volatility on equispaced moneyness grid

# +
print('Fitting SVI model')

# choose the moneyness grid for plot
m_grid = tile(arange(-0.3,0.35,0.05)[np.newaxis,...],(n_, 1))

# Starting guess for SVI parameters
theta_phi_start = namedtuple('theta_phi_start', 'theta4 theta5 theta6')
theta_var_ATM_start = namedtuple('theta_var_ATM_start', 'theta1 theta2 theta3')
theta_phi_start.theta4 = 0
theta_phi_start.theta5 = 0
theta_phi_start.theta6 = 0
theta_var_ATM_start.theta1 = 0
theta_var_ATM_start.theta2 = 0
theta_var_ATM_start.theta3 = 0

# Initialize variables
theta = zeros((6, t_))
sigma_m = zeros((n_, k_, t_))

# SVI fit
for t in range(t_):
    # fit SVI at time t
    theta_var_ATM, theta_phi, _ = FitSigmaSVI(tau, delta, sigma_delta[:,:, t], y, theta_var_ATM_start, theta_phi_start)
    sigma_m[:,:, t] = SigmaSVI(tau, m_grid, y, theta_var_ATM, theta_phi)
    # for the following iteration
    theta_var_ATM_start = theta_var_ATM
    theta_phi_start = theta_phi
    theta[:,t] = r_[theta_var_ATM.theta1, theta_var_ATM.theta2, theta_var_ATM.theta3, theta_phi.theta4, theta_phi.theta5,
                     theta_phi.theta6]

vars_to_save = {varname: var for varname, var in locals().items() if isinstance(var,(np.ndarray,np.float,np.int))}
savemat(os.path.join(TEMPORARY_DB,'db_FitSVI'),vars_to_save)
# -

# ## Plot the fitted implied volatility surface and the evolution of the six parameters

# +
m_grid = m_grid[0]

f, ax = subplots(1,1,subplot_kw={'projection':'3d'})
X, Y = np.meshgrid(m_grid,tau)
ax.plot_surface(X, Y, sigma_m[:,:, t_-1])
ax.view_init(31,-138)
xlabel('Moneyness', labelpad=10)
ylabel('ime to maturity (years)',labelpad=10)
ax.set_zlabel('Volatility (%)',labelpad=10)
ax.set_xlim([min(m_grid), max(m_grid)])
ax.set_ylim([min(tau), max(tau)])
ax.grid(True)
title('SVI fitted implied volatility surface SP500');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

for i in range(2):
    f, ax = subplots(3,1)
    for j, iax in enumerate(ax):
        iax.plot(range(t_), theta[3*i+j,:])
        if j == 0:
            iax.set_title(r'SVI parameters evolution: $\theta_%d$,$\theta_%d$,$\theta_%d$' % (3*i+1,3*i+2,3*i+3))
        iax.set_xlim([1, t_])
        iax.set_ylabel(r'$\theta_%d$' % (3*i+j+1))
        plt.grid(True)

    xlabel('Time')
    plt.tight_layout();
    # save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
