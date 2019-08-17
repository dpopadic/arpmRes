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

# # S_RobustComparMedMeanBDP [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_RobustComparMedMeanBDP&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerBDPMedMean).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

from numpy import ones, zeros, sin, pi, where, percentile, linspace, cov, abs, round, mean, log, tile, array
from numpy import max as npmax
from numpy.random import rand

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, bar, legend, xlim, ylim, subplots, xlabel, yticks

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict
from HistogramFP import HistogramFP
from Price2AdjustedPrice import Price2AdjustedPrice
from GarchResiduals import GarchResiduals
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
t_ = 200

_, x = Price2AdjustedPrice(StocksSPX.Date.reshape(1,-1), StocksSPX.Prices[[25],:], StocksSPX.Dividends[25])  # Cisco Systems Inc
date = StocksSPX.Date[1:]

x = x[[0],-t_:]
date = date[-t_:]
# -

# ## Compute the invariants using GARCH(1,1) fit

epsi = GarchResiduals(x)

# ## Perturb the dataset and compute mean and median

# +
k_ = 3  # number of static figures
sigma2_hist = cov(epsi)
threshold = sigma2_hist*1.7
p_hist = ones((1, t_)) / t_  # historical probabilities
n_bins = int(round(20*log(epsi.shape[1])))

option = namedtuple('option', 'n_bins')
option.n_bins = n_bins
hgram_hist, x_hist = HistogramFP(epsi, p_hist, option)  # historical histogram
mu_hist = mean(epsi, 1)[0]  # historical mean
m_hist = HFPquantile(epsi, array([[0.5]])).squeeze()  # historical median

change = where(abs(epsi - m_hist) > threshold)  # selection of observations to be changed

u_1 = rand(1, t_)
u_2 = rand(1, t_)
u_3 = linspace(2*pi, 4*pi, k_)
epsi_bdp = tile(epsi, (k_, 1))
hgram_bdp = zeros((k_, n_bins))
x_bdp = zeros((k_, n_bins+1))
mu_bdp = zeros((k_, 1))
m_bdp = zeros((k_, 1))
for k in range(k_):
    # shift observations
    epsi_bdp[k, change[1]] = epsi[change]+u_1[change]*abs(epsi[change]-m_hist) * sin(u_2[change]*u_3[k])
    # compute histogram, mean and median from the new dataset
    [hgram_bdp[k,:], x_bdp[k, :]] = HistogramFP(epsi_bdp[[k],:], p_hist, option)  # bdp histogram
    mu_bdp[k] = mean(epsi_bdp[[k], :], 1)  # bdp mean
    m_bdp[k] = HFPquantile(epsi_bdp[[k], :], array([[0.5]]))  # bdp median
# -

# ## Generate static figures showing how the sample mean sensibly varies by perturbing a portion of the dataset, while the median remains the same

# +
colhist = [.8, .8, .8]
c2 = [.05, .45, .7]
Ymax = max(1.1*npmax(hgram_bdp),1.1*npmax(hgram_hist))
Xlim = [percentile(epsi, 100 * 0.007), percentile(epsi, 100 * 0.9965)]
# for k in range(k_):
k = 0

# figure settings
f,ax = subplots(3,1)
# histogram
plt.sca(ax[0])
b1 = bar(x_bdp[k, :-1], hgram_bdp[k, :],width=x_bdp[k, 1]-x_bdp[k,0], facecolor=colhist,edgecolor= [0.65, 0.65, 0.65])
b2 = bar(x_hist[:-1], hgram_hist[0],width=x_hist[1]-x_hist[0], edgecolor=[.06, .31, .75], facecolor='none',lw=1)
xlim(Xlim)
ylim([0, Ymax])
yticks([])
l = legend(['shifting observations','historical'])
# perturbed observations plot
plt.sca(ax[1])
plot(Xlim, [0,0], color='k',lw=.2)
obs_hist = plot(epsi_bdp[k,:], zeros(t_), markersize=4,markeredgecolor= [.6, .6, .6], markerfacecolor= [.6,.6,.6],
                marker='o',linestyle='none')
obs_shifted = plot(epsi_bdp[k, change], zeros(len(change)), markersize=2,markeredgecolor= [.3, .3, .3],
                   markerfacecolor= [.3, .3, .3], marker='o',linestyle='none')
mean_plot = plot([mu_bdp[k], mu_bdp[k]], [0, 0.4], color= [.9, .3, 0], lw=5)
median_plot = plot([m_bdp[k], m_bdp[k]], [-0.4, 0], color = c2, lw = 5)
xlim(Xlim)
ylim([-0.5, 1])
xlabel('Shifted observations',color='k')
qT3 = 'sample mean =  % 3.1f x 10$^{-2}$'%(mu_bdp[k]*10**2)
qT4 = 'sample median =  % 3.1f x 10$^{-2}$'%(m_bdp[k]*10**2)
plt.text(Xlim[1], 0.7, qT3, color= [.9, .3, 0],horizontalalignment='right',verticalalignment='top')
plt.text(Xlim[1], 0.9, qT4, color=c2,horizontalalignment='right',verticalalignment='top')
leg = legend(handles=[obs_hist[0], obs_shifted[0], mean_plot[0], median_plot[0]],
             labels=['Hist. obs.','Shifted obs.','Sample mean','Sample med.'], loc='upper left',ncol=2)
# historical observations plot
plt.sca(ax[2])
plot(Xlim, [0,0], color='k',lw=.2)#
plot(epsi, zeros((1, t_)),markersize=4,markeredgecolor='b',markerfacecolor='b',marker='o',linestyle='none')
plot([mu_hist, mu_hist], [0, 0.4], color= [.9, .3, 0], lw=5)
plot([m_hist, m_hist], [-0.4, 0], color=c2, lw=5)
xlim(Xlim)
ylim([-0.5, 1])
xlabel('Historical observations',color='k')
qT1 = 'sample mean =  % 3.1f x 10$^{-2}$'%(mu_hist*10**2)
qT2 = 'sample median =  % 3.1f x 10$^{-2}$'%(m_hist*10**2)
plt.text(Xlim[1], 0.7, qT1, color= [.9, .3, 0],horizontalalignment='right',verticalalignment='top')
plt.text(Xlim[1], 0.9, qT2, color=c2,horizontalalignment='right',verticalalignment='top')
# histogram
plt.sca(ax[0])
b1 = bar(x_bdp[k, :-1], hgram_bdp[k, :], width=x_bdp[k, 1]-x_bdp[k, 0], facecolor=colhist,edgecolor= [0.65, 0.65, 0.65])
b2 = bar(x_hist[:-1], hgram_hist[0], width=x_hist[1]-x_hist[0], edgecolor=[.06, .31, .75], facecolor='none',lw=1)
xlim(Xlim)
ylim([0, Ymax])
yticks([])
l = legend(['shifting observations','historical'])
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
