#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This script considers the projected P&L of a call options, where the log-underlying is modeled with a GARCH(1,1)
# process and the yield and log-implied volatility are modeled as a VAR(1)
# and computes its first and second order Taylor approximations.
# The results are printed in Figure 1, while the call option P&L as a function of the underlying value
# is plotted in Figure 2.
# -

# ## For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-call-option-pltaylor).

# +
# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from collections import namedtuple

import numpy as np
from numpy import arange, reshape, ones, zeros, std, \
    where, diff, round, mean, log, exp, sqrt, tile, r_
from numpy import min as npmin, max as npmax

from scipy.stats import norm
from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, scatter, ylabel, \
    xlabel, title

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, interpne, struct_to_dict
from RollPrices2YieldToMat import RollPrices2YieldToMat
from Delta2MoneynessImplVol import Delta2MoneynessImplVol
from FitVAR1 import FitVAR1
from intersect_matlab import intersect
from HistogramFP import HistogramFP
from FitGARCHFP import FitGARCHFP
from FPmeancov import FPmeancov
from PerpetualAmericanCall import PerpetualAmericanCall
from NormalScenarios import NormalScenarios

# -

# ## Upload the implied-volatility and the underlying value data from db_ImpliedVol_SPX
# and the realized time series of the rolling values contained in db_SwapCurve.

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_ImpliedVol_SPX'))
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_ImpliedVol_SPX'))
try:
    db_swap = loadmat(os.path.join(GLOBAL_DB, 'db_SwapCurve'))
except FileNotFoundError:
    db_swap = loadmat(os.path.join(TEMPORARY_DB, 'db_SwapCurve'))

db_ImpliedVol_SPX = struct_to_dict(db['db_ImpliedVol_SPX'])
DF_Rolling = struct_to_dict(db_swap['DF_Rolling'])

# ## run S_PricingCallOptionValue which computes the call option exact pricing

from S_PricingCallOptionValue import *

# call option price at time t=0
d1_0 = (log(S_u[0, 0] / k) / sqrt(expiry / 252) + sqrt(expiry / 252) * (
            Y[0, 0] + 0.5 * exp(LogSigma_interp[0, 0]) ** 2)) / exp(LogSigma_interp[0, 0])
d2_0 = d1_0 - exp(LogSigma_interp[0, 0]) * sqrt(expiry / 252)
Vcall_0 = S_u[0, 0] * (norm.cdf(d1_0) - exp(
    -((log(S_u[0, 0] / k) / sqrt(expiry / 252)) * sqrt(expiry / 252) + Y[0, 0] * expiry / 252)) * norm.cdf(d2_0))

# call option P&L
PLC_u = Vcall_u - Vcall_0
PLMuC_u = mean(PLC_u, axis=0, keepdims=True)
PLSigmaC_u = std(PLC_u, axis=0, keepdims=True)
# -

# ## Compute the numerical Greeks, in particular the numerical theta, delta, vega,
# ## rho, gamma, vanna and volga.

# +
# effective theta
time = r_[horiz_u[0], horiz_u[0] + dt]
PL_dt = zeros((1, 2))

for i in range(2):
    tau = (expiry - time[i]) / 252
    Moneyness = log(S_u[0, 0] / k) / sqrt(tau)
    LogSigma_interp_d = interpne(LogSigma_u[:, :, 0, 0], r_[tau, Moneyness], [maturity, m_grid])
    d1 = (Moneyness + sqrt(tau) * (Y[0, 0] + 0.5 * exp(LogSigma_interp_d) ** 2)) / exp(LogSigma_interp_d)
    d2 = d1 - exp(LogSigma_interp_d) * sqrt(tau)
    PL_dt[0, i] = S_u[0, 0] * (norm.cdf(d1) - exp(-(Moneyness * sqrt(tau) + Y[0, 0] * tau)) * norm.cdf(d2))

theta_eff = (PL_dt[0, 1] - PL_dt[0, 0]) / dt

# effective delta and effective gamma

ds = 1  # underlying increment
underl_value = r_[S_u[0, 0] + ds, S_u[0, 0] - ds, S_u[0, 0]]
tau = (expiry - horiz_u[0]) / 252
PL_ds = zeros((1, 3))
PL_dss = zeros((1, 3))

for i in range(3):
    underlying = underl_value[i]
    Moneyness = log(underlying / k) / sqrt(tau)
    LogSigma_interp_d = interpne(LogSigma_u[:, :, 0, 0], r_[tau, Moneyness], [maturity, m_grid])
    d1 = (Moneyness + sqrt(tau) * (Y[0, 0] + 0.5 * exp(LogSigma_interp_d) ** 2)) / exp(LogSigma_interp_d)
    d2 = d1 - exp(LogSigma_interp_d) * sqrt(tau)
    PL_ds[0, i] = underlying * (norm.cdf(d1) - exp(-(Moneyness * sqrt(tau) + Y[0, 0] * tau)) * norm.cdf(d2))

delta_eff = (PL_ds[0, 0] - PL_ds[0, 1]) / (2 * ds)
gamma_eff = (PL_ds[0, 0] - 2 * PL_ds[0, 2] + PL_ds[0, 1]) / (ds ** 2)

# effective vega and effective volga

dsig = 0.01  # sigma increment
PL_dsig = zeros((1, 3))
dsigma = r_[dsig, -dsig, 0]

for i in range(3):
    Moneyness = log(S_u[0, 0] / k) / sqrt(tau)
    LogSigma_interp_d = interpne(LogSigma_u[:, :, 0, 0] + dsigma[i], r_[tau, Moneyness], [maturity, m_grid])
    d1 = (Moneyness + sqrt(tau) * (Y[0, 0] + 0.5 * exp(LogSigma_interp_d) ** 2)) / exp(LogSigma_interp_d)
    d2 = d1 - exp(LogSigma_interp_d) * sqrt(tau)
    PL_dsig[0, i] = S_u[0, 0] * (norm.cdf(d1) - exp(-(Moneyness * sqrt(tau) + Y[0, 0] * tau)) * norm.cdf(d2))

vega_eff = (PL_dsig[0, 0] - PL_dsig[0, 1]) / (2 * dsig)
volga_eff = (PL_dsig[0, 0] - 2 * PL_dsig[0, 3 - 1] + PL_dsig[0, 1]) / (dsig ** 2)

# effective rho
dy = 0.001  # yield increment
yield_ = r_[Y[0, 0] - dy, Y[0, 0] + dy]
PL_dy = zeros((1, 2))

for i in range(2):
    Moneyness = log(S_u[0, 0] / k) / sqrt(tau)
    LogSigma_interp_d = interpne(LogSigma_u[:, :, 0, 0], r_[tau, Moneyness], [maturity, m_grid])
    d1 = (Moneyness + sqrt(tau) * (yield_[i] + 0.5 * exp(LogSigma_interp_d) ** 2)) / exp(LogSigma_interp_d)
    d2 = d1 - exp(LogSigma_interp_d) * sqrt(tau)
    PL_dy[0, i] = S_u[0, 0] * (norm.cdf(d1) - exp(-(Moneyness * sqrt(tau) + yield_[i] * tau)) * norm.cdf(d2))

rho_eff = (PL_dy[0, 1] - PL_dy[0, 0]) / (2 * dy)

# effective vanna
PL_pdsig = zeros((1, 2))
PL_mdsig = zeros((1, 2))

for i in range(2):
    underlying = underl_value[i]
    Moneyness = log(underlying / k) / sqrt(tau)

    LogSigma_interp1 = interpne(LogSigma_u[:, :, 0, 0] + dsig, r_[tau, Moneyness], [maturity, m_grid])
    d1 = (Moneyness + sqrt(tau) * (Y[0, 0] + 0.5 * exp(LogSigma_interp1) ** 2)) / exp(LogSigma_interp1)
    d2 = d1 - exp(LogSigma_interp1) * sqrt(tau)
    PL_pdsig[0, i] = underlying * (norm.cdf(d1) - exp(-(Moneyness * sqrt(tau) + Y[0, 0] * tau)) * norm.cdf(d2))

    LogSigma_interp2 = interpne(LogSigma_u[:, :, 0, 0] - dsig, r_[tau, Moneyness], [maturity, m_grid])
    d1 = (Moneyness + sqrt(tau) * (Y[0, 0] + 0.5 * exp(LogSigma_interp2) ** 2)) / exp(LogSigma_interp2)
    d2 = d1 - exp(LogSigma_interp2) * sqrt(tau)
    PL_mdsig[0, i] = underlying * (norm.cdf(d1) - exp(-(Moneyness * sqrt(tau) + Y[0, 0] * tau)) * norm.cdf(d2))

vanna_eff = (PL_pdsig[0, 0] - PL_mdsig[0, 0] - PL_pdsig[0, 1] + PL_mdsig[0, 1]) / (4 * ds * dsig)
# -

# ## Compute the first order Taylor approximation and the second order Taylor approximation
# ## of the call option P&L.

# +
delta_t = tile((horiz_u[:index + 1] - horiz_u[0]).reshape(1, -1), (j_, 1))[:, -1]  # time changes
delta_y = Y[:, index + 1] - Y[0, 0]  # yield changes
delta_s = S_u[:, index + 1] - S_u[0, 0]  # underlying changes
delta_sig = exp(LogSigma_interp[:, -1]) - exp(LogSigma_interp[0, 0])  # implied volatility changes

# first order approx
Taylor_first = theta_eff * delta_t + delta_eff * delta_s + vega_eff * delta_sig + rho_eff * delta_y
# second order approx
Taylor_second = theta_eff * delta_t + delta_eff * delta_s + vega_eff * delta_sig + rho_eff * delta_y + 0.5 * gamma_eff * delta_s ** 2 + vanna_eff * (
            delta_s * delta_sig) + 0.5 * volga_eff * delta_sig ** 2

# Plot a few simulated paths of the call option P&L up to the selected horizon (6 months),
# along with the first order approximation, the second order approximation and the exact pricing
# of the call option P&L. Plot also the mean and the standard deviation of the projected P&L distribution.
# Furthermore show the scatter plot of the call option P&L as a function of the underlying value.

lgrey = [0.8, 0.8, 0.8]  # light grey
dgrey = [0.4, 0.4, 0.4]  # dark grey
lblue = [0.27, 0.4, 0.9]  # light blu
orange = [0.94, 0.35, 0]  # orange
j_sel = 14  # selected MC simulations

figure()
# simulated path, mean and standard deviation
plot(horiz_u[:index + 1], PLC_u[:j_sel, :index + 1].T, color=lgrey)
l1 = plot(horiz_u[:index + 1], PLMuC_u[0, :index + 1], color='g')
l2 = plot(horiz_u[:index + 1], PLMuC_u[0, :index + 1] + PLSigmaC_u[0, :index + 1], color='r')
plot(horiz_u[:index + 1], PLMuC_u[0, :index + 1] - PLSigmaC_u[0, :index + 1], color='r')
# histogram
option = namedtuple('option', 'n_bins')
option.n_bins = round(10 * log(j_))
y_hist, x_hist = HistogramFP(PLC_u[:, [index + 1]].T, pp_, option)
scale = 0.8 * PLSigmaC_u[0, index + 1] / npmax(y_hist)
y_hist = y_hist * scale
shift_y_hist = horiz_u[index + 1] + y_hist

emp_pdf = plt.barh(x_hist[:-1], shift_y_hist[0] - horiz_u[index + 1], left=horiz_u[index + 1],
                   height=x_hist[1] - x_hist[0], facecolor=lgrey, edgecolor=lgrey, lw=2)  # empirical pdf

# first order approximation
y_hist1, x_hist1 = HistogramFP(np.atleast_2d(Taylor_first), pp_, option)
y_hist1 = y_hist1 * scale
shift_T_first = horiz_u[index + 1] + y_hist1
l3 = plot(shift_T_first[0], x_hist1[:-1], color=lblue)

# second order approximation
y_hist2, x_hist2 = HistogramFP(np.atleast_2d(Taylor_second), pp_, option)
y_hist2 = y_hist2 * scale
shift_T_second = horiz_u[index + 1] + y_hist2
l4 = plot(shift_T_second[0], x_hist2[:-1], color=orange)
legend(handles=[l1[0], l2[0], emp_pdf[0], l3[0], l4[0]],
       labels=['mean', '+/- st.deviation', 'horizon pdf', 'first order approx', 'second order approx'])
xlabel('time (days)')
ylabel('Call option P&L')
title('Call option P&L Taylor approximation');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

# scatter plot
figure()
scatter(S_u[:, index + 1], PLC_u[:, index + 1], 3, dgrey, '*')
xlabel('Underlying')
ylabel('Call option P&L')
title('Scatter plot call option P&L vs. underlying');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
