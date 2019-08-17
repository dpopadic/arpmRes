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

# # S_InverseCallTransformSigma [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_InverseCallTransformSigma&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=exer-log-call-impl-vol-copy-1).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

from numpy import arange, array, zeros
from numpy import min as npmin, max as npmax

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, xlim, ylim, subplots, ylabel, \
    xlabel, title

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict
from InverseCallTransformation import InverseCallTransformation
from SigmaSVI import SigmaSVI
from FitSigmaSVI import FitSigmaSVI

# parameters
y = 0  # risk free rate
m = 0  # selected moneyness
tau = 1  # selected maturity
# -

# ## Upload data from db_ImpliedVol_SPX

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_ImpliedVol_SPX'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_ImpliedVol_SPX'),
                 squeeze_me=True)  # implied volatility surface for SP500

db_ImpliedVol_SPX = struct_to_dict(db['db_ImpliedVol_SPX'])

tau_db = db_ImpliedVol_SPX.TimeToMaturity
delta = db_ImpliedVol_SPX.Delta  # delta-moneyness
sigma_delta = db_ImpliedVol_SPX.Sigma

t_ = sigma_delta.shape[2]
# -

# ## For each observation, use function FitSigmaSVI to compute the SVI parameters and function SigmaSVI to compute the volatility

# +
# Starting guess for SVI parameters
theta_phi_start = namedtuple('theta_phi_start', 'theta4 theta5 theta6')
theta_phi_start.theta4 = 0
theta_phi_start.theta5 = 0
theta_phi_start.theta6 = 0

theta_var_ATM_start = namedtuple('theta_var_ATM_start', 'theta1 theta2 theta3')
theta_var_ATM_start.theta1 = 0
theta_var_ATM_start.theta2 = 0
theta_var_ATM_start.theta3 = 0

# initialize variables
sigma_m = zeros((1, t_))

# SVI fit
for t in range(t_):
    # fit SVI at time t
    theta_var_ATM, xpar_phi, _ = FitSigmaSVI(tau_db, delta, sigma_delta[:,:,t], y, theta_var_ATM_start, theta_phi_start)
    sigma_m[0,t] = SigmaSVI(array([tau]), array([[m]]), y, theta_var_ATM, xpar_phi)
# for the following iteration
theta_var_ATM_start = theta_var_ATM
theta_phi_start = xpar_phi
# -

# ## Compute the inverse-call-implied volatility, using function InvCallTransformation

print('Performing the inverse-call transformation')
# choose the parameter for inverse call function
eta = 0.25
invcsigma_m = InverseCallTransformation(sigma_m, {1:eta})

# ## Plot the inverse-call-implied volatility evolution and the inverse-call transformation

# +
f, ax = subplots(1,2)

# inverse-call implied volatility evolution
plt.sca(ax[0])
plot(arange(t_), invcsigma_m[0])
xlabel('Time')
ylabel('inverse-call implied volatility')
xlim([1, t_])
ylim([npmin(invcsigma_m), npmax(invcsigma_m)])
title('inverse-call implied volatility time series')
plt.xticks([])
plt.grid(True)

# inverse-call transformation
plt.sca(ax[1])
plot(sigma_m.T, invcsigma_m.T)
xlabel('$\sigma$')
ylabel('$c^{-1}$($\sigma$)')
xlim([npmin(sigma_m), npmax(sigma_m)])
ylim([npmin(invcsigma_m), npmax(invcsigma_m)])
title('Inverse call transformation')
plt.grid(True)
plt.tight_layout();
plt.show()
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
