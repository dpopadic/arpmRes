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

# # S_CopulaMarginalDistribution [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_CopulaMarginalDistribution&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerCopulaMargDist).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import arange, array, ones, zeros, sort, argsort, diff, diag, abs, log, exp, sqrt, r_
from numpy import sum as npsum, min as npmin, max as npmax

from scipy.stats import t
from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, xlim, scatter, ylabel, \
    xlabel, title, xticks
import matplotlib.dates as mdates

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict, date_mtop
from intersect_matlab import intersect
from ConditionalFP import ConditionalFP
from BootstrapNelSieg import BootstrapNelSieg
from Tscenarios import Tscenarios
from FactorAnalysis import FactorAnalysis
from MaxLikelihoodFPLocDispT import MaxLikelihoodFPLocDispT
from CopMargComb import CopMargComb

# parameters
par_start = namedtuple('par','theta1 theta2 theta3 theta4_squared')
par_start.theta1 = 0.05
par_start.theta2 = 0.05
par_start.theta3 = 0.05
par_start.theta4_squared = 0.05  #
tau = array([0.0833,1,2,3,4,5,6,7,8,9,10,15,20,25,30])  # Starting values and time to maturities for NS parameters time series extraction
tau_HL = 80  # Half life parameter (days)
nu = 4  # degrees of freedom of the t copula we want to fit
nu_vec = arange(2,31)
nu_ = len(nu_vec)
j_ = 2000  # number of scenarios
k_ = 1  # factors for correlation shrinkage
# -

# ## Upload databases

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_Stocks'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_Stocks'), squeeze_me=True)

SPX = struct_to_dict(db['SPX'])

try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_CorporateBonds'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_CorporateBonds'), squeeze_me=True)

GE = struct_to_dict(db['GE'])

try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_VIX'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_VIX'), squeeze_me=True)

VIX = struct_to_dict(db['VIX'])
# -

# ## Compute the time series of daily S&P500 index's returns and extract the daily time series of VIX

# +
#S&P 500
ret_SP500 = diff(log(SPX.Price_close))
DateSP = SPX.Date[1:]

# Conditioning variable: VIX
DateVIX = VIX.Date
vix = VIX.value
# -

# ## Compute the time series of daily increments of the Nielson-Siegel parameters of the spot yield curve for the GE bond

# +
# Bond schedule
b_sched_GE = zeros((max(GE.Coupons.shape[0],GE.Expiry_Date.shape[0]),2))
b_sched_GE[:, 0] = GE.Coupons/100
b_sched_GE[:, 1] = GE.Expiry_Date

# prices
b_GE = GE.Dirty_Prices/100

# NS parameters' daily increments time series

t_ = len(GE.Date)
thetaGE = zeros((4, t_))
thetaGE[0], thetaGE[1], thetaGE[2], thetaGE[3], *_ = BootstrapNelSieg(GE.Date, b_GE, b_sched_GE, tau, par_start)
DateGE = GE.Date
# -

# ## Match the observations in the three datasets

# +
date, idx_sp, idx_GE = intersect(DateSP, DateGE)
ret_SP500 = ret_SP500[idx_sp]
thetaGE = thetaGE[:, idx_GE]
dates, I_sp_ge, I_vix = intersect(date, DateVIX)
ret_SP500 = ret_SP500[I_sp_ge]
thetaGE = thetaGE[:, I_sp_ge]
vix = vix[I_vix]

epsi = r_[ret_SP500[np.newaxis,...], thetaGE]
i_, t_ = epsi.shape
# -

# ## Compute the Flexible Probabilities conditioned on VIX

# +
# prior
lam = log(2) / tau_HL
prior = exp(-lam*abs(arange(t_, 1 + -1, -1))).reshape(1,-1)
prior = prior / npsum(prior)

# conditioner
conditioner = namedtuple('conditioner', ['Series', 'TargetValue', 'Leeway'])
conditioner.Series = vix.reshape(1,-1)
conditioner.TargetValue = array([[vix[-1]]])
conditioner.Leeway = 0.3
# Flexible Probabilities
p = ConditionalFP(conditioner, prior)
# ## Fit the t copula
# ## estimate marginal distributions by fitting a Student t distribution via
# ## MLFP and recover the invariants' grades
u = zeros((i_, t_))
epsi= sort(epsi, 1)  # We sort scenario in ascending order (in order to apply CopMargComb later)
for i in range(i_):
    mu_nu = zeros(nu_)
    sig2_nu = zeros(nu_)
    like_nu = zeros(nu_)
    for k in range(nu_):
        nu_k = nu_vec[k]
        mu_nu[k], sig2_nu[k],_ = MaxLikelihoodFPLocDispT(epsi[[i],:], p, nu_k, 10 ** -6, 1)
        epsi_t = (epsi[[i], :] - mu_nu[k]) / sqrt(sig2_nu[k])
        like_nu[k] = npsum(p * log(t.pdf(epsi_t, nu_k) / sqrt(sig2_nu[k])))  # likelihood
        j_nu = argsort(like_nu)[::-1]
        # take as estimates the parameters giving rise to the highest likelihood
    nu_marg = max(nu_vec[j_nu[0]], 10)
    mu_marg = mu_nu[j_nu[0]]
    sig2_marg = sig2_nu[j_nu[0]]
    u[i, :] = t.cdf((epsi[i, :] - mu_marg) / sqrt(sig2_marg), nu_marg)
# Map the grades into standard Student t realizations
epsi_tilde = zeros((i_, t_))
for i in range(i_):
    epsi_tilde[i,:] = t.ppf(u[i, :], nu)

# fit the ellipsoid via MLFP

mu, sigma2,_ = MaxLikelihoodFPLocDispT(epsi_tilde, p, nu, 10 ** -6, 1)

# Shrink the correlation matrix toward a low-rank-diagonal structure
rho2 = np.diagflat(diag(sigma2) ** (-1 / 2))@sigma2@np.diagflat(diag(sigma2) ** (-1 / 2))
rho2,*_ = FactorAnalysis(rho2, array([[0]]), k_)
rho2 = np.real(rho2)
# -

# ## Generate scenarios from the estimated t copula

optionT = namedtuple('option', 'dim_red stoc_rep')
optionT.dim_red = 0
optionT.stoc_rep = 0
tcop_scen = Tscenarios(nu, zeros((i_, 1)), rho2, j_, optionT)

# ## Compute the copula-marginal distribution scenarios

grades_MC = t.cdf(tcop_scen, nu)
epsi_MC = CopMargComb(epsi, u, grades_MC)

# ## FIGURE

# +
figure()
# scatter plot
scatter(100*epsi_MC[0], epsi_MC[1], s=10, c=[0.6, 0.6, 0.6], marker='*')
title('COPULA-MARGINAL Distribution')
xlabel('S&P 500 daily return (%)')
ylabel('NS first parameter')
plt.axis([100*npmin(epsi_MC[0])- 0.01, 100*npmax(epsi_MC[0]) + 0.01, npmin(epsi_MC[1]) - 0.01, npmax(epsi_MC[1])+0.01])
# vix plot
date_xtick = arange(99, len(vix), 380)
dates_dt = array([date_mtop(i) for i in dates])
xticklab = dates_dt[date_xtick];
myFmt = mdates.DateFormatter('%d-%b-%Y');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

figure()
ax1 = plt.gca()
plt.bar(dates_dt, p[0], width=dates_dt[1].toordinal()-dates_dt[0].toordinal(),color= 'grey')
ax1.xaxis.set_major_formatter(myFmt)
ax2 = ax1.twinx()
ax2.plot(dates_dt, vix, color=[0, 0, 0.6],lw=1)
ax2.plot(dates_dt, conditioner.TargetValue[0]*ones(t_),color='r',linestyle='--',lw=1)
ax1.set_xlim([min(dates_dt), max(dates_dt)])
ax1.set_xticks(xticklab)
ax1.set_ylim([0, 1.1*npmax(p)])
ax1.set_yticks([])
ax2.set_yticks(arange(20,100,20))
ax2.set_ylim([npmin(vix), 1.1*npmax(vix)])
ax2.set_ylabel('VIX',color=[0, 0, 0.6])
title('Flexible Probabilities');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

