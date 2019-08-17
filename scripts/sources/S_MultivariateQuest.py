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

# # S_MultivariateQuest [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_MultivariateQuest&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-chain-multiv-quest).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import reshape, ones, zeros, diff, linspace, log, sqrt, tile, r_
from numpy import min as npmin, max as npmax

from scipy.stats import norm
from scipy.io import loadmat

import matplotlib.pyplot as plt

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import struct_to_dict
from intersect_matlab import intersect
from Price2AdjustedPrice import Price2AdjustedPrice
from RollPrices2YieldToMat import RollPrices2YieldToMat
from Delta2MoneynessImplVol import Delta2MoneynessImplVol
from FitVAR1 import FitVAR1
from ExponentialDecayProb import ExponentialDecayProb
from InverseCallTransformation import InverseCallTransformation
# -

# ## Upload databases and match the time series of interest to work with synchronous observations

# +
#load
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_Stocks'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_Stocks'), squeeze_me=True)

StocksSPX = struct_to_dict(db['StocksSPX'], as_namedtuple=False)

try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_SwapCurve'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_SwapCurve'), squeeze_me=True)

DF_Rolling = struct_to_dict(db['DF_Rolling'], as_namedtuple=False)

try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_ImpliedVol_SPX'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_ImpliedVol_SPX'), squeeze_me=True)

db_ImpliedVol_SPX = struct_to_dict(db['db_ImpliedVol_SPX'], as_namedtuple=False)

# merge
[common, i_stocks, i_rates] = intersect(StocksSPX['Date'], DF_Rolling['Dates'])
[common, i_others, i_options] = intersect(common, db_ImpliedVol_SPX['Dates'])
StocksSPX['Date'] = StocksSPX['Date'][i_stocks[i_others]]
StocksSPX['Prices'] = StocksSPX['Prices'][:, i_stocks[i_others]]
DF_Rolling['Dates'] = DF_Rolling['Dates'][i_rates[i_others]]
DF_Rolling['Prices'] = DF_Rolling['Prices'][:, i_rates[i_others]]
db_ImpliedVol_SPX['Dates'] = db_ImpliedVol_SPX['Dates'][i_options]
db_ImpliedVol_SPX['Underlying'] = db_ImpliedVol_SPX['Underlying'][i_options]
db_ImpliedVol_SPX['Sigma'] = db_ImpliedVol_SPX['Sigma'][:,:, i_options]

# len of the time series
t_riskdrivers = len(common)
# -

# ## 1a-1b Quest for invariance
# ## Stocks: compute the log-adjusted values and obtain the invariants, i.e. the compounded returns, as their increments

# +
Stocks = namedtuple('Stocks', 'n_ x_tnow d_ epsi i_')

Stocks.n_ = 7  # we consider the first 7 stocks in the dataset
# Risk drivers: log-adjust prices
x_stocks = zeros((Stocks.n_,t_riskdrivers))
for n in range(Stocks.n_):
    x_stocks[n,:] = log(Price2AdjustedPrice(StocksSPX['Date'].reshape(1,-1), StocksSPX['Prices'][[n], :], StocksSPX['Dividends'][n])[0])

Stocks.x_tnow = x_stocks[:,-1]  # current value of the risk drivers
Stocks.d_ = Stocks.n_  # number of risk drivers for stocks

# Invariants: compounded returns (increments in the risk drivers)
Stocks.epsi = diff(x_stocks, 1, 1)  # past realizations of the invariants
Stocks.i_ = Stocks.n_  # number of invariants for stocks
t_ = Stocks.epsi.shape[1]  # len of the time series of the invariants
# -

# ## Zero Coupon Bonds: compute the shadow rates, fit a VAR(1) model and obtain the invariants as residuals

# +
Bonds = namedtuple('Bonds', 'n_ epsi i_ RatingProj I_D V_thor ')
Rates = namedtuple('Rates', 'tau eta x_tnow d_ alpha beta')

Bonds.n_ = 5  # number of bonds

# Risk drivers: shadow rates. They are assumed to follow a VAR(1) process
Rates.tau = DF_Rolling['TimeToMat'][2:]  # time to maturity of the key rates
Rates.eta = 0.013  # parameter for inverse call transformation
shadowrates = InverseCallTransformation(RollPrices2YieldToMat(Rates.tau, DF_Rolling['Prices'][2:,:])[0], {1:Rates.eta})  # shadow rates = InverseCallTransformation((yields to maturity, eta))
Rates.x_tnow = shadowrates[:,-1]  # current value of the risk drivers
Rates.d_ = shadowrates.shape[0]  # number of risk drivers

# Fit a VAR(1) model to the historical series of the shadow rates
# dx = diff(shadowrates, 1, 2)
dt = 1  # 1 day
p = ones((1, t_)) / t_  # flat flexible probabilities
Rates.alpha, Rates.beta, *_ = FitVAR1(shadowrates, p, 5, 0, 0, 0)
# [Rates.alpha, Rates.beta] = FitVAR1(dx, shadowrates((:,1:-1),p, 5, 0, 0, 0))
# [Rates.beta, Rates.alpha] = FitVAR1MVOU(dx, shadowrates((:,1:-1), dt, p, 5, 0, 0, 0, 0, 'VAR1'))

# Invariants: residuals of the VAR(1) model
Bonds.epsi = shadowrates[:, 1:]-tile(Rates.alpha[...,np.newaxis], (1, t_))-Rates.beta@shadowrates[:, : -1]
Bonds.i_ = Bonds.epsi.shape[0]  # number of invariants
# -

# ## Options: compute the m-moneyness

# +
Options = namedtuple('Options', 'n_ x_tnow d epsi i_')
SPX = namedtuple('SPX', 'n_ x_tnow d epsi i_')
ImplVol = namedtuple('SPX', 'tau n_tau')

Options.n_ = 3  # number of options
# Risk drivers: the log-value of the S&P 500 and the log-implied volatility surface follow a random walk the
# short-shadow rate has been modeled above as risk driver for bonds
# Log-value of the underlying (S&P500)
x_SPX = log(db_ImpliedVol_SPX['Underlying'])
SPX.x_tnow = x_SPX[-1]
SPX.epsi = diff(x_SPX)  # invariants

# Log-implied volatility
ImplVol.tau = db_ImpliedVol_SPX['TimeToMaturity']  # the time to expiry grid
ImplVol.n_tau = len(ImplVol.tau)  # len of the time to expiry grid

delta = db_ImpliedVol_SPX['Delta']  # delta-moneyness grid
k_ = len(delta)  # len of the delta-moneyness grid

sigma_delta = db_ImpliedVol_SPX['Sigma']  # implied volatility (delta-moneyness parametrization)

# Short rate
shortrate,_ = RollPrices2YieldToMat(Rates.tau[0], DF_Rolling['Prices'][[0]])
y0_grid_t = zeros((ImplVol.n_tau,k_,t_riskdrivers))
for t in range(t_riskdrivers):
    y0_grid_t[:,:,t] = tile(shortrate[0,[t]], (ImplVol.n_tau, k_))

# Compute the m-parametrized log-implied volatility surface and reshape it to a 2-dimensional matrix
# Moneyness grid
max_moneyness = npmax(tile(norm.ppf(delta)[np.newaxis,...,np.newaxis], (ImplVol.n_tau, 1, t_riskdrivers))*sigma_delta -
                      (y0_grid_t + sigma_delta ** 2 / 2)* tile(sqrt(ImplVol.tau)[...,np.newaxis,np.newaxis],
                                                               (1, k_, t_riskdrivers)))*0.8
min_moneyness = npmin(tile(norm.ppf(delta)[np.newaxis,...,np.newaxis], (ImplVol.n_tau, 1, t_riskdrivers))*sigma_delta
                      - (y0_grid_t + sigma_delta ** 2 / 2)* tile(sqrt(ImplVol.tau)[...,np.newaxis,np.newaxis],
                                                                 (1, k_, t_riskdrivers)))*0.8
ImplVol.n_moneyness = 6
ImplVol.moneyness_grid = linspace(min_moneyness, max_moneyness, ImplVol.n_moneyness)

# For each observation, use function Delta2MoneynessImplVol to pass from the delta-parametrized to the m-parametrized implied volatility surface
ImplVol.s2 = zeros((ImplVol.n_tau, ImplVol.n_moneyness, t_riskdrivers))
# initialization
for t in range(t_riskdrivers):
    for n in range(ImplVol.n_tau):
        ImplVol.s2[n,:,t] = Delta2MoneynessImplVol(sigma_delta[n,:, t], delta, ImplVol.tau[n], y0_grid_t[n,:, t], ImplVol.moneyness_grid)[0]

log_implVol = log(reshape(ImplVol.s2, (ImplVol.n_tau*(ImplVol.n_moneyness),
                          t_riskdrivers),'F'))  # reshaped log implied volatility surface
ImplVol.x_tnow = log_implVol[:,-1]
ImplVol.epsi = diff(log_implVol, 1, 1)

# Invariants
Options.epsi = r_[SPX.epsi.reshape(1,-1), ImplVol.epsi]
Options.i_ = Options.epsi.shape[0]  # number of invariants
Options.d = log_implVol.shape[0] + 1  # number of risk drivers: entries of the log-impl vol and log-underlying
# -

# ## Invariants (all) and Exponential decay probabilities

epsi = r_[Stocks.epsi, Bonds.epsi, Options.epsi]
i_, t_ = epsi.shape
p = ExponentialDecayProb(t_, 250)

# ## Current value of (all) the risk drivers

x_tnow = r_[Stocks.x_tnow, Rates.x_tnow, SPX.x_tnow, ImplVol.x_tnow]
d_ = x_tnow.shape[0]
n_ = Stocks.n_ + Bonds.n_ + Options.n_
