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

# # S_NumRiskPropagationHeston [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_NumRiskPropagationHeston&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-heston-num-risk-prop).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import arange, zeros, diff, abs, log, exp, sqrt, r_
from numpy import sum as npsum

from scipy.io import loadmat

import matplotlib.pyplot as plt

import sympy
from sympy import symbols, I

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import struct_to_dict
from FPmeancov import FPmeancov
from FitCIR_FP import FitCIR_FP
from HestonChFun_symb import HestonChFun_symb as HestonChFun
# -

# ## Upload databases

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_Stocks'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_Stocks'), squeeze_me=True)

SPX = struct_to_dict(db['SPX'])
# -

# ## Compute the time series of risk driver

date = SPX.Date
x = log(SPX.Price_close)
dx = diff(x)

# ## Estimate realized variance

# +
s_ = 252  # forward/backward parameter
lambda1 = log(2) / 5  # half-life one week
p1 = exp(-lambda1*abs(arange(-s_,s_+1))).reshape(1,-1)
p1 = p1 / npsum(p1)

t_var = len(dx) - 2*s_
y = zeros((1, t_var))
for s in range(t_var):
    dx_temp = dx[s:s + 2*s_+1]
    y[0,s] = p1@(dx_temp.T**2) # daily variance

dx = dx[s_:s_ + t_var]
x = x[s_:s_ + t_var]
# -

# ## Calibrate the CIR process

# +
t_obs = 252*4  # 4 years
lambda2 = log(2) / (21*9)  # half-life 9 months
p2 = exp(-lambda2*arange(t_obs, 1 + -1, -1)).reshape(1,-1)
p2 = p2 / npsum(p2)

delta_t = 1  # fix the unit time-step to 1 day

par_CIR = FitCIR_FP(y[0,-t_obs:], delta_t, None, p2)

kappa = par_CIR[0]
y_ = par_CIR[1]
eta = par_CIR[2]
# -

# ## Estimate mu (drift parameter of X) and rho (correlation between Brownian motions)

# +
dy = diff(y)
xy = r_[dx[-t_obs:].reshape(1,-1), dy[:,-t_obs:]]
[mu_xy, sigma2_xy] = FPmeancov(xy, p2)  # daily mean vector and covariance matrix

mu = mu_xy[0]  # daily mean
rho = sigma2_xy[0, 1] / sqrt(sigma2_xy[0, 0]*sigma2_xy[1, 1])  # correlation parameter
# -

# ## Compute analytical variance at horizon tau via characteristic function

# +
omega, x1, x2, x3, x4, x5, x6, x7, tau = symbols('omega x1 x2 x3 x4 x5 x6 x7 tau')

f = HestonChFun(omega / I, x1, x2, x3, x4, x5, x6, x7, tau)
mu1 = sympy.diff(f, omega, 1)
mu2 = sympy.diff(f, omega, 2)
#
# # symbolic conditional variance
sigma2_tau_sym = mu2.subs([(omega,0)]) - mu1.subs([(omega,0)])**2
#
# # numerical conditional variance as a function of horizon tau
sigma2_tau = sigma2_tau_sym.subs({x1: mu[0], x2: kappa, x3: y_, x4: eta, x5: rho, x6: x[-1], x7: y[0,-1]})
