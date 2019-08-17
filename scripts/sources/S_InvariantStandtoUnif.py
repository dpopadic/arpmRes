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

# # S_InvariantStandtoUnif [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_InvariantStandtoUnif&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-stand-to-unif-vue).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

from numpy import arange, zeros, argsort, log, exp, sqrt, tile
from numpy import sum as npsum

from scipy.stats import t
from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, ylim, scatter, ylabel, \
    xlabel, title

plt.style.use('seaborn')

from ARPM_utils import save_plot
from CONFIG import GLOBAL_DB, TEMPORARY_DB
from HistogramFP import HistogramFP
from MaxLikelihoodFPLocDispT import MaxLikelihoodFPLocDispT

# Parameters
tau_HL = 80
nu_vec = arange(2, 31)
nu_ = len(nu_vec)
# -

# ## Upload database

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_zcbInvariants'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_zcbInvariants'), squeeze_me=True)

epsi = db['epsi']

[i_, t_] = epsi.shape
# -

# ## For each marginal invariant, estimate the location and dispersion parameters through MLFP

# +
mu_MLFP = zeros((1, i_))
sig2_MLFP = zeros((1, i_))
nu_MLFP = zeros((1, i_))

# flexible probabilities
lam = log(2) / tau_HL
p = exp((-lam * arange(t_, 1 + -1, -1))).reshape(1, -1)
p = p / npsum(p)

# estimate marginal distributions
for i in range(i_):

    mu_nu = zeros((1, nu_))
    sig2_nu = zeros((1, nu_))
    like_nu = zeros((1, nu_))
    for j in range(nu_):
        nu = nu_vec[j]
        mu_nu[0, j], sig2_nu[0, j], _ = MaxLikelihoodFPLocDispT(epsi[[i], :], p, nu, 10 ** -6, 1)
        epsi_t = (epsi[i, :] - mu_nu[0, j]) / sqrt(sig2_nu[0, j])
        like_nu[0, j] = sum(p[0] * log(t.pdf(epsi_t, nu) / sqrt(sig2_nu[0, j])))  # Log-likelihood

    j_nu = argsort(like_nu[0])[::-1]
    nu_MLFP[0, i] = nu_vec[j_nu[0]]
    mu_MLFP[0, i] = mu_nu[0, j_nu[0]]
    sig2_MLFP[0, i] = sig2_nu[0, j_nu[0]]  # Take as estimates the one giving rise to the highest log-likelihood
# -

# ## Recover the time series of standardized uniform variables

u = zeros((i_, t_))
for i in range(i_):
    u[i, :] = t.cdf((epsi[i, :] - mu_MLFP[0, i]) / sqrt(sig2_MLFP[0, i]), nu_MLFP[0, i])

# ## Compute the histograms

p = tile(1 / t_, (1, t_))  # flat probabilities
option = namedtuple('option', 'n_bins')
option.n_bins = 2 * log(t_)
[f_u1, x_u2] = HistogramFP(u[[0]], p, option)
[f_u2, x_u1] = HistogramFP(u[[1]], p, option)

# ## Generate the figure

# +
# ## Generate the figure

f, ax = plt.subplots(2, 3, figsize=(10, 5))
fsize = 8
x = arange(1, t_ + 1)
# scatter plots
plt.sca(ax[0, 0])
h1 = scatter(x, epsi[0], 10, [0.5, 0.5, 0.5], '*')
title('SP500 residuals', fontsize=fsize)
xlabel('Time', fontsize=fsize)
ylabel('Residuals', fontsize=fsize)
ylim([min(epsi[0]) - 0.1, max(epsi[0]) + 0.1])
plt.sca(ax[1, 0])
h2 = scatter(x, epsi[1], 10, [0.5, 0.5, 0.5], '*')
title('Shadow rate residuals', fontsize=fsize)
xlabel('Time', fontsize=fsize)
ylabel('Residuals', fontsize=fsize)
ylim([min(epsi[1]) - 0.0001, max(epsi[1]) + 0.0001])
plt.sca(ax[0, 1])
h3 = scatter(x, u[0], 10, [0.5, 0.5, 0.5], '*')
title('Standardized uniform SP500 residuals', fontsize=fsize)
xlabel('Time', fontsize=fsize)
ylabel('Residuals', fontsize=fsize)
ylim([min(u[0]) - 0.1, max(u[0]) + 0.1])
plt.sca(ax[1, 1])
h4 = scatter(x, u[1], 10, [0.5, 0.5, 0.5], '*')
title('Standardized uniform shadow rate residuals', fontsize=fsize)
xlabel('Time', fontsize=fsize)
ylabel('Residuals', fontsize=fsize)
ylim([min(u[1]) - 0.1, max(u[1]) + 0.1])
# histograms
plt.sca(ax[0, 2])
ax[0, 2].ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
plt.barh(x_u1[:-1], f_u1[0] / t_, height=x_u1[1] - x_u1[0], facecolor=[0.7, 0.7, 0.7], edgecolor=[0.5, 0.5, 0.5])
ylim([min(u[0]) - 0.1, max(u[0]) + 0.1])
title('Histogram standardized uniform SP500 residuals', fontsize=fsize)
plt.sca(ax[1, 2])
ax[1, 2].ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
plt.barh(x_u2[:-1], f_u2[0] / t_, height=x_u2[1] - x_u2[0], facecolor=[0.7, 0.7, 0.7], edgecolor=[0.5, 0.5, 0.5])
ylim([min(u[1]) - 0.1, max(u[1]) + 0.1])
title('Histogram standardized uniform s. rate residuals', fontsize=fsize)
plt.tight_layout(h_pad=1, w_pad=0.1);
# # save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
#
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

