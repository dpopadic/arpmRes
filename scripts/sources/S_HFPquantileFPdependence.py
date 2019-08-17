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

# # S_HFPquantileFPdependence [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_HFPquantileFPdependence&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerHFPquantilePlot).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

from numpy import arange, array, ones, zeros, percentile, linspace, round, log, r_
from numpy import min as npmin, max as npmax
from numpy.random import randn

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, bar, legend, xlim, ylim, ylabel, \
    yticks
import matplotlib.dates as mdates

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict, date_mtop
from HistogramFP import HistogramFP
from Price2AdjustedPrice import Price2AdjustedPrice
from GarchResiduals import GarchResiduals
from BlowSpinFP import BlowSpinFP
from HFPquantile import HFPquantile
# -

# ## Upload the database

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_Stocks'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_Stocks'), squeeze_me=True)

StocksSPX = struct_to_dict(db['StocksSPX'])
# -

# ## Compute the dividend-adjusted returns of one stock

# +
t_ = 300
_, x = Price2AdjustedPrice(StocksSPX.Date.reshape(1,-1), StocksSPX.Prices[[25],:], StocksSPX.Dividends[25])  # Cisco Systems Inc
date = StocksSPX.Date[1:]

x = x[[0],-t_:]
date = date[-t_:]
# -

# ## Compute the invariants using GARCH(1,1) fit

epsi = GarchResiduals(x)

# ## Compute the Flexible Probability profiles using Blow-Spin method

epsi_BlowSpin = r_[epsi, randn(1, t_)]  # random generation of dataset's second row
b = 2  # number of blows
s = 3  # number of spins
p, ens = BlowSpinFP(epsi_BlowSpin, b, s)

# ## Compute the hist-histogram, hist-quantile, HFP-histogram and HFP-quantile

# +
# k_ = b+s
k_ = 5
options = namedtuple('option', 'n_bins')
options.n_bins = round(10*log(t_))  # number of bins for the histograms
p_quant = arange(0,1.005,0.005).reshape(1,-1)  # quantile probability levels
p_flat = ones((1, t_)) / t_
# historical histogram
hgram_hist, x_bin = HistogramFP(epsi, p_flat, options)
# historical quantile
q_hist = percentile(epsi, 100*p_quant.flatten())

hgram_HFP = zeros((k_, int(options.n_bins)))
q_HFP = zeros((k_, p_quant.shape[1]))
for k in range(k_):
    # HFP-histogram
    [hgram_HFP[k, :], _] = HistogramFP(epsi, p[[k],:], options)
    # HFP-quantile
    q_HFP[k, :] = HFPquantile(epsi, p_quant, p[[k], :])
# -

# ## Generate some figures showing how the HFP-quantile and the HFP-histogram evolve as the FP profile changes

# +
hfp_color = [.9, .5, 0.5]
date_dt = array([date_mtop(i) for i in date])
myFmt = mdates.DateFormatter('%d-%b-%Y')

for k in range(k_):

    f,ax = plt.subplots(3,1)
    P = p[[k],:]

    # quantile plot
    plt.sca(ax[0])
    xlim([0, 1])
    ylim([npmin(epsi) - 0.1, npmax(epsi) + 0.2])
    plot(p_quant[0], q_hist, color='b')
    plot(p_quant[0], q_HFP[k, :], color= hfp_color)
    leg0 = legend(['historical quantile','HFP-quantile'])

    # histogram plot
    plt.sca(ax[1])
    b = bar(x_bin[:-1], hgram_HFP[k, :], width=x_bin[1]-x_bin[0], facecolor=hfp_color,edgecolor='k', label='HFP')
    b1 = bar(x_bin[:-1], hgram_hist[0], width=x_bin[1]-x_bin[0], edgecolor='b',facecolor='none',label='historical')
    yticks([])
    l = legend()

    # Flexible Probabilities profile
    plt.sca(ax[2])
    b = bar(date_dt,P[0], width=date_dt[1].toordinal()-date_dt[0].toordinal(),facecolor= [.7, .7, .7], edgecolor=[.7, .7, .7])
    d = linspace(0,t_-1,4,dtype=int)
    xtick = date_dt[d]
    xlim([min(date_dt), max(date_dt)])
    plt.gca().xaxis.set_major_formatter(myFmt)
    ylim([0, npmax(P)])
    yticks([])
    ylabel('FP')
    ensT = 'Effective Num.Scenarios =  % 3.0f'%ens[0,k]
    plt.text(date_dt[10], npmax(P) - npmax(P) / 10, ensT, horizontalalignment='left',verticalalignment='bottom')
    plt.tight_layout();
    # save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
