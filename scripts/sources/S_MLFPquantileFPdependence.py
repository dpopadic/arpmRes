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

# # S_MLFPquantileFPdependence [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_MLFPquantileFPdependence&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerMLquantPlot).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

from numpy import arange, array, ones, zeros, sort, where, ceil, round, log, r_, linspace, max as npmax
from numpy.random import randn

from scipy.stats import genpareto
from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, bar, legend, xlim, ylim, ylabel, \
    title
import matplotlib.dates as mdates

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import struct_to_dict, date_mtop
from HistogramFP import HistogramFP
from Price2AdjustedPrice import Price2AdjustedPrice
from GarchResiduals import GarchResiduals
from BlowSpinFP import BlowSpinFP
from QuantileGenParetoMLFP import QuantileGenParetoMLFP
from FitGenParetoMLFP import FitGenParetoMLFP
from HFPquantile import HFPquantile
# -

# ## Upload the database

try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_Stocks'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_Stocks'), squeeze_me=True)

# ## Compute the dividend-adjusted returns of one stock

# +
t_ = 600

StocksSPX = struct_to_dict(db['StocksSPX'])

_, x = Price2AdjustedPrice(StocksSPX.Date.reshape(1,-1), StocksSPX.Prices[[25], :], StocksSPX.Dividends[25])  # Cisco Systems Inc
date = StocksSPX.Date[1:]

x = x[[0],-t_:]
date = date[-t_:]
# -

# ## Compute the invariants using GARCH(1,1) fit

epsi = GarchResiduals(x, p0=[0, 0.01, 0.5])

# ## Compute the Flexible Probability profiles using Blow-Spin method

epsi_BlowSpin = r_[epsi, randn(1, t_)]  # random generation of dataset's second row
b = 2  # number of blows
s = 3  # number of spins
p, ens = BlowSpinFP(epsi_BlowSpin, b, s, spinscale=.7)

# ## Estimate quantiles and tail approximation using the EVT/MLFP method

# +
k_ = b + s
p_bar = 0.1  # probability threshold
p_quant = r_[arange(10**-4, p_bar+10**-4,10 ** -4), arange(p_bar+0.001,1.001,0.001)]  # quantile probability levels

q_hist = HFPquantile(epsi, p_quant.reshape(1,-1))
epsi_bar = q_hist[0,p_quant == p_bar][0]  # threshold

# data below the threshold
l_1 = where(epsi < epsi_bar)[1]
l_2 = where(p_quant <= p_bar)[0]
epsi_ex = epsi_bar - epsi[0,l_1]  # dataset of the conditional excess distribution

# MLFP quantile and Generalized Pareto Distribution
q_MLFP = zeros((k_, len(l_2)))
f_MLFP = zeros((k_, len(l_1)))
for k in range(k_):
    csi_MLFP, sigma_MLFP = FitGenParetoMLFP(epsi_ex, p[k, l_1])  # Maximum Likelihood optimization with Generalized Pareto Distribution
    f_MLFP[k, :] = genpareto.pdf(sort(epsi_ex), c=0, scale=csi_MLFP, loc=sigma_MLFP-1)

    q_MLFP[k, :] = QuantileGenParetoMLFP(epsi_bar, p_bar, csi_MLFP, sigma_MLFP, p_quant[l_2])[0]  # MLFP-quantile

# historical quantile below the threshold
q_bt = q_hist[0,l_2]
# histogram of the pdf of the Conditional Excess Distribution
t_ex_ = len(epsi_ex)
options = namedtuple('options', 'n_bins')
options.n_bins = round(12 * log(t_ex_))
hgram_ex, x_bin = HistogramFP(epsi_ex.reshape(1,-1), ones((1, t_ex_)) / t_ex_, options)
# -

# ## Generate figures showing the difference between the historical data and the EVT/MLFP estimations

for k in range(k_):
    f = figure()
    date_dt = array([date_mtop(i) for i in date])
    myFmt = mdates.DateFormatter('%d-%b-%Y')
    # quantile plot
    ax = plt.subplot2grid((5, 1), (0, 0), rowspan=2)
    xlim([0, npmax(p_quant[l_2])])
    plot(p_quant[l_2], q_bt, color='b')
    plot(p_quant[l_2], q_MLFP[k, :], color='r')
    ylim([q_bt[0] - 5, 0.8 * epsi_bar])
    legend(['historical quantile', 'EVT / MLFP - quantile'])
    th = 'threshold:  % 3.2f' % epsi_bar
    plt.text(0.001, 1, th, horizontalalignment='left')
    title('QUANTILE')
    # conditional excess distribution and Generalized Pareto fit
    ax = plt.subplot2grid((5, 1), (2, 0), rowspan=2)
    ex = bar(x_bin[:-1], hgram_ex[0], width=x_bin[1] - x_bin[0], edgecolor='b', facecolor="none")
    gpd = plot(sort(epsi_ex), f_MLFP[k, :], color='r')
    plt.axis([0, npmax(epsi_ex), 0, 1.5 * npmax(hgram_ex)])
    legend(handles=[ex[0], gpd[0]], labels=['historical pdf', 'EVT / MLFP pdf'])
    title('CONDITIONAL EXCESS DISTRIBUTION')
    # Flexible Probability profile
    ax = plt.subplot2grid((5, 1), (4, 0))
    bar(date_dt, p[k, :], width=date_dt[1].toordinal() - date_dt[0].toordinal(), facecolor=[.7, .7, .7],
        edgecolor=[.7, .7, .7])
    d = linspace(0, t_ - 1, 4, dtype=int)
    xlim([min(date_dt), max(date_dt)])
    plt.xticks(date_dt[d])
    myFmt = mdates.DateFormatter('%d-%b-%y')
    plt.gca().xaxis.set_major_formatter(myFmt)
    y_lim = ylim()
    ylabel('FP')
    ensT = 'Effective Num.Scenarios =  % 3.0f' % ens[0, k]
    plt.text(date_dt[10], y_lim[1], ensT, horizontalalignment='left', verticalalignment='bottom')
    plt.tight_layout();
