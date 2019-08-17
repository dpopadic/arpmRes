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

# # S_LognormalInnovation [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_LognormalInnovation&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=EBInnovaLogNorm).

# ## Prepare the environment

# +
import os.path as path
import sys, os

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

from numpy import ones, linspace, round, log, exp, sqrt, r_, array
import numpy as np

from scipy.stats import norm

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, bar, legend, xlim, scatter, ylabel, \
    xlabel

plt.style.use('seaborn')

from ARPM_utils import save_plot
from FPmeancov import FPmeancov
from PlotTwoDimEllipsoid import PlotTwoDimEllipsoid
from HistogramFP import HistogramFP
from NormalScenarios import NormalScenarios
from NormInnov import NormInnov

# inputs
j_ = 10000  # simulations
mu = array([[0.95], [0.65]])  # expectation
svec = array([[0.68], [0.46]])  # volatilities
rho = 0.95  # correlation
s2 = np.diagflat(svec)@array([[1, rho], [rho, 1]])@np.diagflat(svec)  # covariance matrix
# -

# ## Generate bivariate lognormal draws

Y = exp(NormalScenarios(mu, s2, j_, 'Riccati')[0])
X = Y[[0]]
Z = Y[[1]]

# ## Compute the sample of innovation

Psi = NormInnov(log(r_[X, Z]), mu, svec, rho)
p = ones((1, j_)) / j_
mu_ZPsi, s2_ZPsi = FPmeancov(r_[Z, Psi], p)  # expectation and covariance of Z and Psi

# ## Visualize empirical pdf of innovation

# +
nbins = round(7*log(j_))
figure()

p = ones((1, Psi.shape[1])) / Psi.shape[1]
option = namedtuple('option', 'n_bins')

option.n_bins = nbins
[n, psi] = HistogramFP(Psi, p, option)
bar(psi[:-1], n[0], width=psi[1]-psi[0], facecolor=[.8, .8, .8],edgecolor='k',label='empirical pdf of $\Psi$')

psimax = max(psi)
psimin = min(psi)
psigrid = linspace(psimin, psimax, 100)
f = norm.pdf(psigrid, mu_ZPsi[1], sqrt(s2_ZPsi[1, 1]))
plot(psigrid, f, color='m', lw= 2,label='pdf of standard normal')
legend()
xlim([psimin, psimax]);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
# -

# ## Display scatter plot and location-dispersion ellipsoids

figure()
scatter(Z, Psi, 0.5, [.5, .5, .5], '*')
PlotTwoDimEllipsoid(mu_ZPsi, s2_ZPsi, 2, 1, [], 'r', 2)
xlabel('Z')
ylabel('$\Psi$');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
