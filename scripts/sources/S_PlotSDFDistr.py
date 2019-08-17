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

# # S_PlotSDFDistr [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_PlotSDFDistr&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-sdfcomparison).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import arange, ones, zeros, diag, eye, exp, sqrt, tile, diagflat
from numpy import sum as npsum, min as npmin, max as npmax
from numpy.linalg import solve
from numpy.random import multivariate_normal as mvnrnd

from scipy.stats import norm, uniform

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, xlim, ylim

plt.style.use('seaborn')

from ARPM_utils import save_plot
from SDFkern import SDFkern

# parameter

# parameters
n_ = 250
j_ = 500
r_rf = 0.05
a_p = 0.7
b_p = 1
a_SDF = 0
b_SDF = 0.9
rho = 0.7
# -

# ## Generate the payoff matrix

# +
# Generate the normal vector
c2 = rho*ones((n_, n_)) + (1 - rho)*eye(n_)  # correlation matrix
X = mvnrnd(zeros(n_), c2, j_).T

# Generate the payoffs
V_payoff = ones((n_, j_))
V_payoff[1] = exp(X[1]) / (sqrt(exp(1) - 1)*exp(0.5))
V_payoff[2::2,:] = (exp(X[2::2,:])-exp(0.5) / (sqrt(exp(1) - 1))*exp(0.5))
V_payoff[3::2,:] = (-exp(-X[3::2,:])+exp(0.5) / (sqrt(exp(1) - 1))*exp(0.5))
V_payoff[2:,:] = diagflat(uniform.rvs(loc=0.8, scale=0.2, size=(n_ - 2, 1)))@V_payoff[2:,:]  # rescaling
V_payoff[2:,:] = V_payoff[2:,:]+tile(uniform.rvs(loc=-0.3, scale=1, size=(n_ - 2, 1)), (1, j_))  # shift
# -

# ## Compute the probabilities

p = uniform.rvs(loc=a_p, scale=b_p-a_p, size=(j_, 1))
p = p /npsum(p)

# ## Compute the "true" Stochastic Discount Factor vector of scenarios

SDF_true = uniform.rvs(loc=a_SDF, scale=b_SDF-a_SDF, size=(1, j_))
c = 1 / ((SDF_true@p)*(1 + r_rf))
SDF_true = SDF_true*c  # constraint on the expectation of SDF

# ## Compute the current values vector

v_tnow = V_payoff@diagflat(p)@SDF_true.T

# ## Compute the projection Stochastic Discount Factor

SDF_proj = v_tnow.T@(solve(V_payoff@diagflat(p)@V_payoff.T,V_payoff))

# ## Compute the Kernel Stochastic Discount Factor

SDF_kern = SDFkern(V_payoff, v_tnow, p)

# ## Generate the figure

# +
# Compute the gaussian smoothed histograms
bw = 0.1  # band-width
x = arange(npmin(SDF_true) - 5*bw,npmax(SDF_true) + 5*bw,0.01)

# Gaussian smoothings
Y = tile(x, (len(SDF_true), 1)) - tile(SDF_true.T, (1, len(x)))
SDF_true = p.T@norm.pdf(Y, 0, bw)
Y = tile(x, (len(SDF_proj), 1)) - tile(SDF_proj.T, (1, len(x)))
SDF_proj = p.T@norm.pdf(Y, 0, bw)
Y = tile(x, (len(SDF_kern), 1)) - tile(SDF_kern.T, (1, len(x)))
SDF_kern = p.T@norm.pdf(Y, 0, bw)

figure()
plot(x, SDF_true[0])
plot(x, SDF_proj[0], 'g')
plot(x, SDF_kern[0], 'm')
yl = ylim()
plot([v_tnow[0], v_tnow[0]], [0, yl[1]], 'k--')
ylim(yl)
xlim([x[0], x[-1]])
legend(['True SDF','Proj SDF','Kern SDF','Risk Free']);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
