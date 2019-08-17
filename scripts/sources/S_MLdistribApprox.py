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

# # S_MLdistribApprox [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_MLdistribApprox&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerMaxLikConsist).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import ones, linspace, sqrt
from numpy import min as npmin, max as npmax

from scipy.stats import t

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, legend, xlim, ylim, subplots

plt.style.use('seaborn')

from ARPM_utils import save_plot
from MaxLikelihoodFPLocDispT import MaxLikelihoodFPLocDispT

# input parameters
t_ = 500  # number of observations
nu = 3  # degrees of freedom
mu = 0  # location parameter
sigma2 = 2  # square dispersion parameter
sigma = sqrt(sigma2)
threshold = 1e-4
last = 1
# -

# ## Generate the observation of the Student t with 3 degree of freedom, location parameter 0 and dispersion parameter 2

Epsi_std = t.rvs(nu, size=(1, t_))
Epsi = mu + sigma*Epsi_std  # Affine equivariance property
x = linspace(npmin(Epsi_std),npmax(Epsi_std),t_+1)

# ## Compute the Maximum Likelihood location and dispersion parameters

p = (1 / t_)*ones((1, t_))  # probabilities
mu_ML, sigma2_ML,_ = MaxLikelihoodFPLocDispT(Epsi, p, nu, threshold, last)

# ## Compute the Maximum Likelihood pdf

sigma_ML = sqrt(sigma2_ML)
fML_eps = t.pdf((x - mu_ML) / sigma_ML, nu)

# ## Compute the Maximum Likelihood cdf

FML_eps = t.cdf((x - mu_ML) / sigma_ML, nu)

# ## Compute the true  pdf and cdf

f_eps = t.pdf((x - mu) / sigma, nu)

# ## Compute the true cdf

F_eps = t.cdf((x - mu) / sigma, nu)

# ## Display the Maximum Likelihood pdf and overlay the true pdf

# +
orange = [.9, .4, .2]
b = [0, 0.5, 1]

f, ax = subplots(2,1)
plt.sca(ax[0])
# plot the maximum likelihood pdf
plot(x, fML_eps[0], lw=1.5,color=orange)
xlim([npmin(x), npmax(x)])
ylim([0, npmax(fML_eps) + 0.15])

# plot the true pdf
plot(x, f_eps, lw=1.5,color=b)
# -

# ## Display the Maximum Likelihood cdf and overlay the true cdf

# +
plt.sca(ax[1])
# plot the maximum likelihood cdf
plot(x, FML_eps[0], color=orange,lw=1.5)
xlim([npmin(x), npmax(x)])
ylim([0, npmax(F_eps) + 0.15])

# plot the true cdf
plot(x, F_eps, lw=1.5,color=b)
legend(['True','Max Likelihood'])
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
