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

# # S_MarkovChainMonteCarlo [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_MarkovChainMonteCarlo&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerMCMC).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import ones, percentile, linspace, round, mean, log, sqrt
from numpy import max as npmax

from scipy.stats import t
from scipy.integrate import trapz

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, bar, legend, xlim, ylim, title

plt.style.use('seaborn')

from ARPM_utils import save_plot
from HistogramFP import HistogramFP
from MetrHastAlgo import MetrHastAlgo
# -

# ## Define the target distribution

# conditional likelihood
nu = 10
mu_hat = 0
sigma2_hat = 0.1
f_hat =lambda x: t.pdf((x - mu_hat) / sqrt(sigma2_hat), nu) / sqrt(sigma2_hat)  # Student t distribution
# prior distribution
mu_pri = 0.3
sigma2_pri = 0.2
f_pri =lambda x: t.pdf((x - mu_pri) / sqrt(sigma2_pri), 1) / sqrt(sigma2_pri)  # Cauchy distribution

# ## Run the Metropolis-Hastings algorithm

# step 0
j_ = 2500  # len of the sample
theta_0 = 5  # initial guess
# run the algorithm
theta, accept_rate = MetrHastAlgo(f_hat, f_pri, theta_0, j_)

# ## Create figure

# colors
Cpri = [0.2, 0.3, 1]
Cpos = [0.9, 0.3, 0.1]
Csam = [0.1, 0.7, 0.1]
# histogram of simulations
p = ones((1, j_)) / j_
option = namedtuple('option', 'n_bins')
option.n_bins = round(10*log(j_))
f_bin, x_bin = HistogramFP(theta[np.newaxis,...], p, option)
# axis settings
l_ = 200
delta1 = percentile(theta, 100 * 0.005)
delta2 = percentile(theta, 100 * 0.995)
mu_theta = mean(theta)
x_1 = mu_theta - (delta2 - delta1)
x_2 = mu_theta + (delta2 - delta1)
x_pi = linspace(x_1, x_2, l_)
# posterior pdf
pdf_pos = (t.pdf((x_pi - mu_pri) / sqrt(sigma2_pri), 1) / sqrt(sigma2_pri)) * (
t.pdf((x_pi - mu_hat) / sqrt(sigma2_hat), 1) / sqrt(sigma2_hat))
pdf_pos = pdf_pos / trapz(pdf_pos,x_pi)
# prior pdf
pdf_pri = t.pdf((x_pi - mu_pri) / sqrt(sigma2_pri), 1) / sqrt(sigma2_pri)

# ## conditional likelihood

# +
pdf_hat = t.pdf((x_pi - mu_hat) / sqrt(sigma2_hat), nu) / sqrt(sigma2_hat)
y_max = max([npmax(f_bin), npmax(pdf_pos), npmax(pdf_pri), npmax(pdf_hat)])

figure()
h = bar(x_bin[:-1], f_bin[0], width=x_bin[1]-x_bin[0],facecolor= [.8, .8, .8],edgecolor='k',label='simulated distribution')
cl = plot(x_pi, pdf_hat, color=Csam,lw=3,label='conditional likelihood')
cg = plot(x_pi, pdf_pri, color=Cpri,lw=3,label='prior distribution')
tar = plot(x_pi, pdf_pos, lw=5,color=Cpos,label='posterior distribution')
xlim([x_1, x_2])
ylim([0, 1.3*y_max])
title('Markov chain Monte Carlo simulations: Metropolis-Hastings algorithm')
legend();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
