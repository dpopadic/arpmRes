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

# # S_CopulaMarginalEstimationRiskDrivers [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_CopulaMarginalEstimationRiskDrivers&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerCopulaEstim2).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import arange, array, ones, zeros, cumsum, percentile, diff, linspace, diag, eye, abs, log, exp, sqrt, tile, r_
from numpy import sum as npsum
from numpy.linalg import solve

from scipy.stats import t
from scipy.linalg import expm
from scipy.io import loadmat, savemat

import matplotlib.pyplot as plt

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import struct_to_dict, datenum
from FPmeancov import FPmeancov
from intersect_matlab import intersect
from RollPrices2YieldToMat import RollPrices2YieldToMat
from ConditionalFP import ConditionalFP
from MaxLikelihoodFPLocDispT import MaxLikelihoodFPLocDispT
from MMFP import MMFP
from CopMargSep import CopMargSep
from VGpdf import VGpdf
from ParamChangeVG import ParamChangeVG
from VAR1toMVOU import VAR1toMVOU
from FitVAR1 import FitVAR1
from ShiftedVGMoments import ShiftedVGMoments
from FitCIR_FP import FitCIR_FP
from InverseCallTransformation import InverseCallTransformation
# -

# ## Upload the databases and match the time series of interest to work with synchronous observations

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_Stocks'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_Stocks'), squeeze_me=True)

SPX = struct_to_dict(db['SPX'],as_namedtuple=False)

try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_SwapCurve'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_SwapCurve'), squeeze_me=True)

DF_Rolling = struct_to_dict(db['DF_Rolling'],as_namedtuple=False)

try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_OptionStrategy'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_OptionStrategy'), squeeze_me=True)

OptionStrategy = struct_to_dict(db['OptionStrategy'],as_namedtuple=False)

try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_VIX'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_VIX'), squeeze_me=True)

VIX = struct_to_dict(db['VIX'],as_namedtuple=False)

DateOptStrat = array([datenum(i) for i in OptionStrategy['Dates']]).T
common, i_spvix, i_rates = intersect(SPX['Date'], DF_Rolling['Dates'])
SPX['Date'] = SPX['Date'][i_spvix]
SPX['Price_close'] = SPX['Price_close'][i_spvix]
VIX['value'] = VIX['value'][i_spvix]
VIX['Date'] = VIX['Date'][i_spvix]
DF_Rolling['Dates'] = DF_Rolling['Dates'][i_rates]
DF_Rolling['Prices'] = DF_Rolling['Prices'][:, i_rates]
common, i_others, i_options = intersect(common, DateOptStrat)
SPX['Date'] = SPX['Date'][i_others]
SPX['Price_close'] = SPX['Price_close'][i_others]
VIX['value'] = VIX['value'][i_others]
VIX['Date'] = VIX['Date'][i_others]
DF_Rolling['Dates'] = DF_Rolling['Dates'][i_others]
DF_Rolling['Prices'] = DF_Rolling['Prices'][:, i_others]
DateOptStrat = DateOptStrat[i_options]
OptionStrategy['cumPL'] = OptionStrategy['cumPL'][i_options]
t_common = len(common)
# -

# ## Extract the invariants from the Heston process as follows:
# ## - Estimate the realized variance y_{t}  for the S&P500 using backward-forward exponential smoothing

# +
x_HST = log(SPX['Price_close']).reshape(1,-1)
dx_HST = diff(x_HST)

# Estimate the realized variance
s_ = 252  # forward/backward parameter
lambda1_HST = log(2) / (2*252)  # half life 2 years
p_y = exp(-lambda1_HST*abs(arange(-s_,s_+1))).reshape(1,-1)
p_y = p_y / npsum(p_y)

t_ = dx_HST.shape[1] - 2*s_
y = zeros(t_)
for s in range(t_):
    dx_temp = dx_HST[[0],s:s + 2*s_+1]
    y[s] = p_y@(dx_temp.T** 2)
# daily variance

dy = diff(y)
dx_HST = dx_HST[[0],s_:-s_]
x_HST = x_HST[[0],s_:-s_]

# - Fit the CIR process to by FitCIR_FP
t_obs = len(dy)  # time series len
p_HST = ones((1, t_obs)) / t_obs  # flexible probabilities
delta_t = 1  # fix the unit time-step to 1 day
par_CIR = FitCIR_FP(y[1:], delta_t, None, p_HST)
kappa = par_CIR[0]
y_bar = par_CIR[1]
eta = par_CIR[2]

# - Estimate the drift parameter and the correlation coefficient between the Brownian motions by FPmeancov
mu_HST, sigma2_x_HST = FPmeancov(r_[dx_HST[[0],1:], dy.reshape(1,-1)], p_HST)  # daily mean vector and covariance matrix
mu_x_HST = mu_HST[0]  # daily mean
rho_HST = sigma2_x_HST[0, 1] / sqrt(sigma2_x_HST[0, 0]*sigma2_x_HST[1,1])  # correlation parameter

# - Extract the invariants
epsi_x_HST = (dx_HST[[0],1:] - mu_x_HST*delta_t) / sqrt(y[1:])
epsi_y = (dy + kappa*(y[1:]-y_bar)*delta_t) / (eta*sqrt(y[1:]))
epsi_HST = r_[epsi_x_HST, epsi_y.reshape(1,-1)]
# -

# ## Extract the invariants for the MVOU process as follows:
# ## - Compute the 2-year and 7-year yield to maturity by RollPrices2YieldToMat and obtain the corresponding shadow rates by InverseCallTransformation
# ## Select the two-year and seven-year key rates and estimate the MVOU process
# ## parameters using functions FitVAR1 and VAR1toMVOU.

# +
_, _, pick = intersect([2, 7], DF_Rolling['TimeToMat']) # select rates (.T2y','7y')
yields,_ = RollPrices2YieldToMat(DF_Rolling['TimeToMat'][pick], DF_Rolling['Prices'][pick,:])
yields = yields[:, s_ + 1:-s_]
d_ = len(pick)

# - Fit the parameters by FitVAR1 and VAR1toMVOU and extract the invariants
eta_ICT = 0.013
x_MVOU = InverseCallTransformation(yields, {1:eta_ICT})  # select rates ('2y','7y')
lam = log(2) / (21*9)  # half-life: 9 months
p_MVOU = exp(-lam*arange(t_obs, 1 + -1, -1)).reshape(1,-1)
p_MVOU = p_MVOU / npsum(p_MVOU)
[alpha, b, sig2_U] = FitVAR1(x_MVOU, p_MVOU, 4)
    # [alpha, b, sig2_U] = FitVAR1(dx_MVOU, x_MVOU(:, 1:-1),p_MVOU, 4)
mu_MVOU, theta_MVOU, sigma2_MVOU,_ = VAR1toMVOU(alpha, b, sig2_U, delta_t)
# [mu_MVOU, theta_MVOU, sigma2_MVOU] = FitVAR1MVOU(dx_MVOU, x_MVOU(:, 1:-1), delta_t, p_MVOU, 4)
epsi_MVOU = x_MVOU[:, 1:] - expm(-theta_MVOU*delta_t)@x_MVOU[:, : -1] + tile((eye(theta_MVOU.shape[0]) - expm(-theta_MVOU*delta_t))@solve(theta_MVOU,mu_MVOU)[...,np.newaxis], (1, t_obs))
# -

# ## Extract the invariants for the variance gamma process and fit their parametric distribution as follows
# ## - Compute the time series of daily P&L and extract the invariants
# ## cumulative P&L

x_VG = OptionStrategy['cumPL'][s_+1:-s_].reshape(1,-1)

# ## invariants (VG is a Levy process-> random walk -> the invariants are the increments)

# +
epsi_VG = diff(x_VG)

# -Fit the parameters of the VG marginal distribution by MMFP
# initial guess on parameters
mu0 = 1
theta0 = -1
sigma0 = 1
nu0 = 1
par0 = [mu0, theta0, sigma0, nu0]

flat_p = ones((1, t_obs)) / t_obs  # flat flexible probabilities

HFP = namedtuple('HFP', ['FlexProbs','Scenarios'])
HFP.FlexProbs = flat_p
HFP.Scenarios = epsi_VG
par = MMFP(HFP, 'SVG', par0)

mu_vg = par.c
theta_vg = par.theta
sigma_vg = par.sigma
nu_vg = par.nu

# -After switching to the (c,m,g) parametrization, compute the marginal pdf and recover the cdf numerically
[par.c, par.m, par.g] = ParamChangeVG(theta_vg, sigma_vg, nu_vg)  # change the parametrization to compute the pdf

# compute the expectation and variance to fix the grid for the pdf
expectation_vg, variance_vg, _, _ = ShiftedVGMoments(0, theta_vg, sigma_vg, nu_vg, 1)
epsi_grid_vg = linspace(expectation_vg - 4*sqrt(variance_vg), expectation_vg + 4*sqrt(variance_vg), t_obs).reshape(1,-1)

pdf_vg = VGpdf(epsi_grid_vg, par, 1)
cdf_vg = cumsum(pdf_vg / npsum(pdf_vg),axis=1)
shifted_epsi_grid_vg = epsi_grid_vg + mu_vg
# -

# ## Set the Flexible Probabilities (conditioned on VIX via Entropy Pooling)

# +
# conditioner
conditioner = namedtuple('conditioner', ['Series', 'TargetValue', 'Leeway'])
conditioner.Series = VIX['value'][- t_obs:].reshape(1,-1)
conditioner.TargetValue = np.atleast_2d(VIX['value'][-1])
conditioner.Leeway = 0.35
# prior
lam = log(2) / (5*252)  # half life 5y
prior = exp(-lam*abs(arange(t_obs, 1 + -1, -1)))
prior = prior / npsum(prior)

p = ConditionalFP(conditioner, prior)
# -

# ## Collect the extracted invariants in a matrix and fit the copula

# +
# invariants
epsi = r_[epsi_HST, epsi_MVOU, epsi_VG]
i_ = epsi.shape[0]
t_obs = epsi.shape[1]

# Rescale the invariants
q1 = percentile(epsi, 25, axis=1,keepdims=True)
q2 = percentile(epsi, 75, axis=1,keepdims=True)
interq_range = q2 - q1
epsi_rescaled = epsi / tile(interq_range, (1, t_obs))

# STEP 1: Invariants grades
epsi_grid, u_grid, grades = CopMargSep(epsi_rescaled, p)
nu = 4

# STEP [1:] Marginal t
epsi_st = zeros(epsi.shape)
for i in range(i_):
    epsi_st[i,:] = t.ppf(grades[i,:], nu)

# STEP 3: Fit ellipsoid (MLFP ellipsoid under Student t assumption)
Tol = 10 ** -6
mu_epsi, sigma2_epsi,_ = MaxLikelihoodFPLocDispT(epsi_st, p, nu, Tol, 1)

# STEP 4: Shrinkage (we don't shrink sigma2)

# STEP 5: Correlation
c2_hat = np.diagflat(diag(sigma2_epsi) ** (-1 / 2))@sigma2_epsi@np.diagflat(diag(sigma2_epsi) ** (-1 / 2))

# Rescale back the invariants'o the original size
epsi_grid = epsi_grid * tile(interq_range, (1, t_obs))
# -

# ## Marginal distributions: HFP distributions for epsi_HST and epsi_MVOU parametric VG distribution for epsi_VG

# +
marginals_grid = r_[epsi_grid[:4,:], shifted_epsi_grid_vg.reshape(1,-1)]
marginals_cdfs = r_[u_grid[:4,:], cdf_vg]

varnames_to_save = ['d_','marginals_grid','marginals_cdfs','c2_hat','mu_epsi','nu','eta_ICT','x_MVOU','mu_MVOU','theta_MVOU','sigma2_MVOU',
                    'delta_t','kappa','y_bar','eta','mu_x_HST','x_HST','mu_vg','theta_vg','sigma_vg','nu_vg','x_VG','y']
vars_to_save = {varname: var for varname, var in locals().items() if isinstance(var,(np.ndarray,np.float,np.int))}
vars_to_save = {varname:var for varname,var in vars_to_save.items() if varname in varnames_to_save}
savemat(os.path.join(TEMPORARY_DB, 'db_CopulaMarginalRiskDrivers'),vars_to_save)
