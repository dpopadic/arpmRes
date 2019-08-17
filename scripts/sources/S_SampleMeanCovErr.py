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

# # S_SampleMeanCovErr [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_SampleMeanCovErr&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerMeanCovErr).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import ones, zeros, percentile, cov, eye, round, mean, log, tile
from numpy import max as npmax, sum as npsum
from numpy.linalg import norm as linalgnorm
from numpy.random import randn
from numpy.random import multivariate_normal as mvnrnd

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, bar, legend, xlim, ylim, title

plt.style.use('seaborn')

from ARPM_utils import save_plot
from HistogramFP import HistogramFP
# -

# ## Generate scenarios for the estimators and their losses

# +
rho = 0.999  # correlation
i_ = 15  # number of invariants
mu = randn(i_, 1)  # true mean
sigma2 = 5*(rho*ones((i_)) + (1 - rho)*eye(i_))  # true covariance
t_ = 20  # len of time series
j_ = 10 ** 4  # number of simulations

M = zeros((i_, j_))
L_M = zeros((1, j_))
Sigma2 = zeros((i_, i_, j_))
L_Sigma2 = zeros((1, j_))
for j in range(j_):
    I = mvnrnd(mu.flatten(),sigma2,t_).T  # i_ x t_end
    # compute the loss of sample mean
    M[:,j] = mean(I, 1)
    L_M[0,j] = npsum((mu - M[:, [j]]) ** 2)
    # compute the loss of sample covariance
    Sigma2[:,:, j] = cov(I,ddof=1)
    L_Sigma2[0,j] = linalgnorm(sigma2 - Sigma2[:,:, j], ord='fro') ** 2
# -

# ## Compute error, bias and inefficiency of both estimators

# sample mean
E_M = mean(M, 1)
er_M = mean(L_M)
ineff2_M = mean(npsum((M - tile(E_M[...,np.newaxis], (1, j_)))**2,axis=0))
bias2_M = er_M - ineff2_M
# sample covariance
E_Sigma2 = mean(Sigma2, 2)
er_Sigma2 = mean(L_Sigma2)
ineff2_Sigma2 = mean(npsum((Sigma2 - tile(E_Sigma2[...,np.newaxis], (1, 1, j_)))**2,axis=(0,1)))
bias2_Sigma2 = er_Sigma2 - ineff2_Sigma2

# ## Create figures

# +
nbins = round(50*log(j_))
colhist = [.8, .8, .8]
orange = [1, 0.4, 0]
dark = [0.2, 0.2, 0.2]
blue = [0, 0.4, 1]

# sample mean assessment
figure()

p = ones((1, L_M.shape[1])) / L_M.shape[1]
option = namedtuple('option', 'n_bins')
option.n_bins = nbins
L_M_hist, L_M_x = HistogramFP(L_M, p, option)
LOSS = bar(L_M_x[:-1], L_M_hist[0], width=L_M_x[1]-L_M_x[0], facecolor= colhist, edgecolor=  'none')
ERROR = plot([0, er_M], [npmax(L_M_hist)*0.04, npmax(L_M_hist)*0.04], color=dark, lw=5)
BIAS = plot([0, bias2_M], [npmax(L_M_hist)*0.01, npmax(L_M_hist)*0.01], color=orange, lw=5)
INEF = plot([bias2_M, er_M], [npmax(L_M_hist)*0.01, npmax(L_M_hist)*0.01], color=blue, lw=5)
xlim([-npmax(L_M)*0.0025, percentile(L_M, 99 + 0.9*(1 - rho))])
ylim([0, 1.1*npmax(L_M_hist)])
title('Sample mean assessment')
l = legend(handles=[LOSS,ERROR[0], BIAS[0],INEF[0]],labels=['loss','error','bias$^2$' ,'ineff.$^2$'])
COR = 'correlation coeff. = % 3.2f'%rho
plt.text(percentile(L_M, 0.99 + 0.009*(1 - rho)), 0.85*npmax(L_M_hist), COR, color='k',horizontalalignment='left');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

# sample covariance assessment
figure()

L_Sigma2_hist, L_Sigma2_x = HistogramFP(L_Sigma2, p, option)
LOSS = bar(L_Sigma2_x[:-1], L_Sigma2_hist[0],width=L_Sigma2_x[1]-L_Sigma2_x[0], facecolor= colhist, edgecolor=  'none')
ymax = npmax(L_Sigma2_hist)
ERROR = plot([0, er_Sigma2], [npmax(L_Sigma2_hist)*0.04, npmax(L_Sigma2_hist)*0.04], color=dark, lw=5)
BIAS = plot([0, bias2_Sigma2], [npmax(L_Sigma2_hist)*0.01, npmax(L_Sigma2_hist)*0.01], color=orange, lw=5)
INEF = plot([bias2_Sigma2, er_Sigma2], [npmax(L_Sigma2_hist)*0.01, npmax(L_Sigma2_hist)*0.01], color=blue, lw=5)
xlim([-npmax(L_Sigma2)*0.0005, percentile(L_Sigma2, 90 + 9.9*(1 - rho))])
ylim([0, 1.1*npmax(L_Sigma2_hist)])
title('Sample covariance assessment')
l = legend(handles=[LOSS,ERROR[0], BIAS[0],INEF[0]],labels=['loss','error','bias$^2$' ,'ineff.$^2$'])
COR = 'correlation coeff. = % 3.2f'%rho
plt.text(percentile(L_Sigma2, 0.9 + 0.099*(1 - rho)), 0.85*npmax(L_Sigma2_hist), COR, color='k',horizontalalignment='left');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
