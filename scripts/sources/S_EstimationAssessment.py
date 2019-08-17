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

# # S_EstimationAssessment [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_EstimationAssessment&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExEstimAssess).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

from tqdm import trange

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import arange, ones, zeros, sort, argsort, cumsum, percentile, diag, eye, round, mean, log, exp, tile, \
    histogram, array, r_, corrcoef, real, diagflat
from numpy import sum as npsum, max as npmax
from numpy.linalg import eig, norm as linalgnorm

from scipy.stats import norm
from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, bar, xlim, ylim, yticks

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict, matlab_percentile
from intersect_matlab import intersect
from MinRelEntFP import MinRelEntFP
from NormalScenarios import NormalScenarios
from EffectiveScenarios import EffectiveScenarios
from ConditionalFP import ConditionalFP
from FactorAnalysis import FactorAnalysis
from CopMargComb import CopMargComb

# initialize variables
i_ = 25  # number of stocks
t_ = 100  # len of time series
j_ = 500  # number of simulated time series for each k-th DGP [low for speed increase for accuracy]
k_ = 5  # number of perturbed DGP's [low for speed]
h = 0  # DGP whose loss is plotted
# -

# ## Upload databases

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_StocksS_P'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_StocksS_P'), squeeze_me=True)

Data = struct_to_dict(db['Data'])

try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_VIX'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_VIX'), squeeze_me=True)

VIX = struct_to_dict(db['VIX'], as_namedtuple=False)
# -

# ## Compute the stocks' log-returns

# +
dates_x = Data.Dates
x = Data.Prices

# compute the log-returns
epsi = log(x[:i_, 1:]/ x[:i_, : -1])
# conditioning variable (VIX)
z = VIX['value']
dates_z = VIX['Date']
# -

# ## Merge the datasets and select the first t_end observations

# +
[dates, i_x, i_z] = intersect(dates_x, dates_z)

epsi = epsi[:, i_x[:t_]]
z = z[i_z[:t_]].reshape(1,-1)
# -

# ## Estimate the distribution of the invariants

# +
# Correlation matrix
d = zeros((1, i_))
rank = 1  # rank

c2 = np.corrcoef(epsi)
c2, *_ = FactorAnalysis(c2, d, rank)
c2 = real(c2)

# Marginals
# prior
lam = 0.0005
prior = exp(lam*(arange(1,t_+1))).reshape(1,-1)
prior = prior / npsum(prior)

# conditioner
VIX = namedtuple('VIX', 'Series TargetValue Leeway')
VIX.Series = z
VIX.TargetValue = np.atleast_2d(matlab_percentile(z.flatten(), 100 * 0.7))
VIX.Leeway = 0.35

# flexible probabilities conditioned via EP
p = ConditionalFP(VIX, prior)

# effective number of scenarios
typ = namedtuple('type','Entropy')
typ.Entropy = 'Exp'
ens = EffectiveScenarios(p, typ)
# -

# ## Perform estimation assessment: compute errors for each perturbed DGP

# +
# noise
Z = norm.rvs(0, 1, size=[i_, k_])
# compute base case eigenvalues
DiagLambda2, e = eig(c2)
log_lambda2_base = log(DiagLambda2)
# initialize
c2_DGP = {}
p_DGP = {}
y = zeros((i_, t_))
ff = zeros((i_, t_))
C2_hat = zeros((i_, i_, j_))
C2_bar = zeros((i_, i_, j_))
L_hat = zeros(j_)
L_bar = zeros(j_)
er_hat = zeros(k_)
er_bar = zeros(k_)

for k in trange(k_,desc='DGP'):
    # Perturb DGP
    if k == 0:
        c2_DGP[k] = real(c2)
        p_DGP[k] = tile(p, (i_, 1))
    else:
        # perturb correlation matrix
        log_lambda2 = log_lambda2_base + Z[:, k] / 100
        lambda2 = exp(log_lambda2)
        c2_DGP[k] = e@diagflat(lambda2)@e.T
        c2_DGP[k][eye(i_) == 1] = 1
        # perturb marginals
        p_DGP[k] = zeros((i_,t_))
        for i in range(i_):
            a =r_[ones((1, t_)), epsi[[i],:]]
            b = r_[array([[1]]),(p@epsi[[i], :].T)*Z[i, k] / 100]
            p_DGP[k][i, :] = MinRelEntFP(p, None, None, a, b)[0]

    # marginals for CMA-combination
    for i in range(i_):
        y[i, :], idy = sort(epsi[i,:]), argsort(epsi[i,:])
        # f = p_DGP[k][0,idy]
        f = p_DGP[k][i,idy]
        ff[i, :] = cumsum(f)

    for j in range(j_):
        # Randomize time series I
        m, _ = NormalScenarios(zeros((i_, 1)), c2_DGP[k], t_, 'Riccati')
        U1 = norm.cdf(m)
        if npsum(U1==0) >= 1:
            print(k)
        I = CopMargComb(y, ff, U1)

        # Evaluate the correlation estimators
        C2_hat[:,:, j] = corrcoef(I)  # sample correlation
        C2_bar[:,:, j] = real(FactorAnalysis(C2_hat[:,:, j], d, rank)[0])  # shrinkage correlation

        # Compute the losses
        L_hat[j] = linalgnorm(C2_hat[:,:, j]-c2_DGP[k], ord='fro')**2  # sample loss
        L_bar[j] = linalgnorm(C2_bar[:,:, j]-c2_DGP[k], ord='fro')**2  # shrinkage loss

    # Compute errors
    er_hat[k] = mean(L_hat)  # sample error
    er_bar[k] = mean(L_bar)  # shrinkage error

    # store loss's distribution and bias for the selected h-th DGP
    if k == h:
        # histograms
        nbins = int(round(10*log(j_)))
        hgram_hat, x_hat = histogram(L_hat, nbins)
        hgram_hat = hgram_hat / (nbins*(x_hat[1] - x_hat[0]))
        hgram_bar, x_bar = histogram(L_bar, nbins)
        hgram_bar = hgram_bar / (nbins*(x_bar[1] - x_bar[0]))

        # compute bias
        bias_hat = linalgnorm(mean(C2_hat, 2) - c2_DGP[k], ord='fro')
        bias_bar = linalgnorm(mean(C2_bar, 2) - c2_DGP[k], ord='fro')
# -

# ## Compute robust and ensemble errors

# +
# Robust
er_rob_hat = npmax(er_hat)
er_rob_bar = npmax(er_bar)

# Ensemble with equal weigths
er_ens_hat = mean(er_hat)
er_ens_bar = mean(er_bar)
# -

# ## Display results

# +
colhist = [.8, .8, .8]
orange = [1, 0.4, 0]
dark = [0.2, 0.2, 0.2]
blue = [0, 0.4, 1]

M = max(npmax(x_hat), npmax(x_bar))

f, ax = plt.subplots(1, 2)
plt.sca(ax[0])
plt.axis()
# sample correlation
LOSS = bar(x_hat[:-1], hgram_hat,width=x_hat[1]-x_hat[0], facecolor=colhist,edgecolor= colhist,zorder=0)
xlim([0, 1.1*M])
ylim([0, 1.1*npmax(hgram_hat)])
yticks([])  #
plot([0, bias_hat ** 2], [npmax(hgram_hat)*0.01, npmax(hgram_hat)*0.01], color=orange, lw=5,zorder=2)
plot([bias_hat ** 2, er_hat[k]], [npmax(hgram_hat)*0.01, npmax(hgram_hat)*0.01], color=blue, lw=5,zorder=1)
plot([0, er_hat[k]], [npmax(hgram_hat)*0.04, npmax(hgram_hat)*0.04], color=dark, lw=5,zorder=1)
plot([0, 0], [0, 0], color='lightgreen',marker='o',markerfacecolor='g',zorder=3)
# global title
f.suptitle('LOSS DISTRIBUTIONS')
# title
ax[0].set_title('Sample correlation')
S_B = 'Bias$^2$:  % 3.2f'% (bias_hat**2)
plt.text(0.01*M, -0.15*npmax(hgram_hat), S_B, color=orange,horizontalalignment='left')
S_I = 'Ineff$^2$ :  % 3.2f'%(er_hat[k]-bias_hat**2)
plt.text(0.01*M, -0.25*npmax(hgram_hat), S_I, color='b',horizontalalignment='left')
S_E = 'Error:  % 3.2f'%er_hat[k]
plt.text(0.01*M, -0.35*npmax(hgram_hat), S_E, color=dark,horizontalalignment='left')
S_WCE = 'Robust Error:  % 3.2f'%er_rob_hat
plt.text(M, -0.25*npmax(hgram_hat), S_WCE, color='r',horizontalalignment='right')
S_EH = 'Ensemble Error:  % 3.2f'%er_ens_hat
plt.text(M, -0.35*npmax(hgram_hat), S_EH, color='r',horizontalalignment='right')
num = 'Test Data Generating Process:  % 3.0f of %3.0f'%(h+1,k_)
plt.text(0, 1.23*npmax(hgram_hat), num, color='k',horizontalalignment='left')
# shrinkage
plt.sca(ax[1])
bar(x_bar[:-1], hgram_bar, width=x_bar[1]-x_bar[0], facecolor=colhist,edgecolor= colhist,zorder=0)
xlim([0, 1.1*M])
ylim([0, 1.1*npmax(hgram_bar)])
plt.yticks([])
plot([0, bias_bar**2], [npmax(hgram_bar)*0.01, npmax(hgram_bar)*0.01], color=orange, lw=5,zorder=2)
plot([bias_bar**2, er_bar[k]], [npmax(hgram_bar)*0.01, npmax(hgram_bar)*0.01], color=blue, lw=5,zorder=1)
plot([0, er_bar[k]], [npmax(hgram_bar)*0.04, npmax(hgram_bar)*0.04], color=dark, lw=5,zorder=1)
plot([0,0], [0,0], color='lightgreen',marker='o',markerfacecolor='g',zorder=3)
# title
ax[1].set_title('Shrinkage correlation')
B = 'Bias$^2$  % 3.2f'% bias_bar**2
plt.text(0.01*M, -0.15*npmax(hgram_bar), B, color=orange,horizontalalignment='left')
I = 'Ineff$^2$: % 3.2f'%(er_bar[k]-bias_bar**2)
plt.text(0.01*M, -0.25*npmax(hgram_bar), I, color='b',horizontalalignment='left')
E = 'Error:  % 3.2f'%er_bar[k]
plt.text(0.01*M, -0.35*npmax(hgram_bar), E, color=dark,horizontalalignment='left')
WCE = 'Robust Error:  % 3.2f'%er_rob_bar
plt.text(M, -0.25*npmax(hgram_bar), WCE, color='r',horizontalalignment='right')
EH = 'Ensemble Error:  % 3.2f'%er_ens_bar
plt.text(M, -0.35*npmax(hgram_bar), EH, color='r',horizontalalignment='right')
f.subplots_adjust(bottom=0.3,top=0.85);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
