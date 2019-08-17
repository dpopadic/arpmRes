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

# # S_PricingScenarioBased [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_PricingScenarioBased&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-chain-hybrid-pricing-scen).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

from scipy.interpolate import interp1d

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import arange, reshape, array, zeros, where, log, exp, sqrt, tile, r_
from numpy import max as npmax

import warnings
warnings.filterwarnings('ignore')

from scipy.io import savemat

import matplotlib.pyplot as plt

plt.style.use('seaborn')

from CONFIG import TEMPORARY_DB
from ARPM_utils import interpne
from PerpetualAmericanCall import PerpetualAmericanCall
from blsprice import blsprice

# script S_ProjectionHybridDefaultableZCB runs the script that projects the market risk drivers (S_ProjectionBootstrap) and generates scenarios for the default indicators
from S_ProjectionHybridDefaultableZCB import *
# -

# ## Stocks. Compute the scenarios of the ex-ante P&L of the stocks via exact pricing starting from the scenarios of the log-values at the horizon

# current values
Stocks.v_tnow = exp(Stocks.x_tnow)
# values at the horizon
Stocks.V_thor = exp(X_path[:Stocks.i_, -1,:])
# P&L's
Stocks.Pi = Stocks.V_thor - tile(Stocks.v_tnow.reshape(-1,1), (1, j_))

# ## Defaultable Zero Coupon Bonds. Compute the scenarios of the ex-ante market and credit P&L of the defaultable zero-coupon bond starting from the scenarios of the path of the shadow rates and of the default indicators
# ##Note: for simplicity we didn't model the spreads. The yields of the ZCB's are those of the reference curve (for the issuers, regardless of their rating).

# +
# ## current values

Bonds.tau_tnow = array([160, 80, 70, 50, 140]) / 252  # time to maturity of the bonds at tnow
ShadowRates_tnow = X_path[Rates.idx, 0, 0]  # shadow yield curve at tnow
interp = interp1d(Rates.tau, ShadowRates_tnow,fill_value='extrapolate')  # interpolate the curve to obtain the shadow yields for the relevant time to maturity
shadowy_tnow = interp(Bonds.tau_tnow)
y_tnow = PerpetualAmericanCall(shadowy_tnow, {'eta':Rates.eta})  # from shadow yields to yields

Bonds.v_tnow = zeros((Bonds.n_, 1))
for n in range(Bonds.n_):
    Bonds.v_tnow[n]=exp((-Bonds.tau_tnow[n])*y_tnow[n])  # exact pricing function

# exposures at default
Bonds.EAD=zeros((Bonds.n_, j_, tau_proj))

defaulted = {}
for tau in range(tau_proj):
    indi, indj =where(Bonds.I_D[:, tau,:])
    defaulted[tau] = r_['-1',indi.reshape(-1,1),indj.reshape(-1,1)]
    if tau > 0:
        defaulted[tau]=np.setdiff1d(defaulted[tau], defaulted[tau-1], 0)
    if not defaulted[tau]:
        Bonds.tau_tstep=Bonds.tau_tnow-tau / 252  # time to maturity of the bonds at the projection step tau
        ShadowRates_tstep=X_path[Rates.idx, tau+1,:]  # shadow yield curve at the projection step tau
        interp = interp1d(Rates.tau, ShadowRates_tstep.T, fill_value='extrapolate')
        Shadowy_tstep = interp(Bonds.tau_tstep)  # interpolate the curve to obtain the shadow rates for the relevant time to maturity
        Y_thor = PerpetualAmericanCall(Shadowy_tstep, {'eta': Rates.eta})  # from shadow yields to yields
        for n in range(defaulted[tau].shape[0]):
            # exposure at default
            Bonds.EAD[defaulted[tau][n, 0], tau, defaulted[tau][n, 1]]= exp(-Bonds.tau_tstep[defaulted[tau][n, 0]]@Y_thor[defaulted[tau][n, 0], defaulted[tau][n, 1]])

# scenarios for the market and credit value at the horizon
Bonds.recoveryrates=[.6, .6, .5, .4, .7]

Bonds.tau_thor=Bonds.tau_tnow-tau_proj / 252  # time to maturity of the bonds at the projection step tau
ShadowRates_thor=X_path[Rates.idx, tau+1,:]  # shadow yield curve at the projection step tau

interp = interp1d(Rates.tau, ShadowRates_thor.T, fill_value='extrapolate')
Shadowy_thor = interp(Bonds.tau_thor).T  # interpolate the curve to obtain the shadow rates for the relevant time to maturity
Y_thor = PerpetualAmericanCall(Shadowy_thor, {'eta': Rates.eta})  # from shadow yields to yields

Bonds.V_thor = zeros((Bonds.n_,Y_thor.shape[1]))
Bonds.V_mc_thor = zeros((Bonds.n_,Y_thor.shape[1]))

for n in range(Bonds.n_):
    Bonds.V_thor[n,:]= exp(-Bonds.tau_thor[n]*Y_thor[n,:])
    Bonds.V_mc_thor[n,:]=npmax(Bonds.I_D[n,:,:]*Bonds.recoveryrates[n]*Bonds.EAD[n,:,:], 1).T + (1-Bonds.I_D[n,:,-1])*Bonds.V_thor[n, :]

# P&L's
Bonds.Pi = Bonds.V_mc_thor - tile(Bonds.v_tnow, (1, j_))
# -

# ## Pricing: Call options

# +
Options.strikes = array([1100, 1150, 1200])

# Implied volatility paths (reshaped)

implvol_idx = arange(Stocks.i_ + Bonds.i_ + 1,i_)
LogImplVol_path = reshape(X_path[implvol_idx,:,:], (ImplVol.n_tau, ImplVol.n_moneyness, tau_proj + 1, j_),'F')

# current value
Options.tau_tnow = array([30, 30, 30]) / 252  # time to expiry of the options at tnow (days)
shortrate_tnow = PerpetualAmericanCall(Rates.x_tnow[0], {'eta':Rates.eta})
Options.v_tnow = zeros((Options.n_, 1))  # initialize

Moneyness_tnow = zeros((Options.n_,1))
for n in range(Options.n_):
    Moneyness_tnow[n] = log(SPX.x_tnow / Options.strikes[n]) / sqrt(Options.tau_tnow[n])  # Moneyness
    # interpolated log-implied volatility
    logimplVol_interp_tnow = interpne(LogImplVol_path[:,:, 0,0], r_['-1',Options.tau_tnow[n], Moneyness_tnow[n]], [ImplVol.tau, ImplVol.moneyness_grid])
    Options.v_tnow[n] = blsprice(exp(SPX.x_tnow), Options.strikes[n], shortrate_tnow, Options.tau_tnow[n], exp(logimplVol_interp_tnow))

# value at the horizon
SPX_thor = exp(X_path[implvol_idx[0] - 1, -1,:])
Shortrate_thor = PerpetualAmericanCall(X_path[Rates.idx[0], -1,:],{'eta':Rates.eta}).T
Options.tau_thor = Options.tau_tnow - tau_proj / 252  # time to expiry of the options at the horizon
Options.V_thor = zeros((Options.n_,j_))
Moneyness_thor = zeros((Options.n_,j_))
for n in range(Options.n_):
    for j in range(j_):
        Moneyness_thor[n, j] = log(SPX_thor[j] / Options.strikes[n]) / sqrt(Options.tau_thor[n])  # Moneyness
        LogImplVol_interp_thor = interpne(LogImplVol_path[:,:, -1, j], r_[Options.tau_thor[n], Moneyness_thor[n, j]]
        , [ImplVol.tau,ImplVol.moneyness_grid])
        Options.V_thor[n, j] = blsprice(SPX_thor[j], Options.strikes[n], Shortrate_thor[j], Options.tau_thor[n], exp(LogImplVol_interp_thor))

# P&L's
Options.Pi = Options.V_thor - tile(Options.v_tnow, (1, j_))
# -

# ## Current values (all)

v_tnow = r_[Stocks.v_tnow.flatten(), Bonds.v_tnow.flatten(), Options.v_tnow.flatten()]

# ## Ex-ante P&L's scenarios (all)

# +
Pi = r_[Stocks.Pi, Bonds.Pi, Options.Pi]


Stocks = {k:v for k,v in vars(Stocks).items() if not k.startswith('_') and not isinstance(v,property)}
Bonds = {k:v for k,v in vars(Bonds).items() if not k.startswith('_') and not isinstance(v,property)}
Options = {k:v for k,v in vars(Options).items() if not k.startswith('_') and not isinstance(v,property)}
SPX = {k:v for k,v in vars(SPX).items() if not k.startswith('_') and not isinstance(v,property)}

varnames_to_save = ['Stocks', 'Bonds', 'Options', 'SPX', 'SPX_thor', 'n_', 't_', 'j_', 'tau_proj', 'v_tnow', 'Pi', 'p']
vars_to_save = {varname: var for varname, var in locals().items() if isinstance(var, (np.ndarray, np.float, np.int, dict)) and varname in varnames_to_save}
savemat(os.path.join(TEMPORARY_DB, 'db_PricingScenarioBased'), vars_to_save)

