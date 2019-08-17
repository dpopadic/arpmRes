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

# # S_MaximumLikelihood [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_MaximumLikelihood&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-4-ex-mle).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import arange, array, zeros, log, exp, sqrt, r_
from numpy import max as npmax

from scipy.stats import t, lognorm
from scipy.io import loadmat, savemat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, scatter

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot
# -

# ## Upload dataset

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_TimeSeries'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_TimeSeries'), squeeze_me=True)

i_t = db['i_t']
t_ = len(i_t)
# -

# ## Define the set of possible values for theta (approximating [-0.04,-0.01] with a finite set of points)

theta_set = r_[arange(-.04,-.009,0.001), array([0.02]), array([0.03])]

# ## compute the log-likelihood for each value of theta in theta_set

loglikelihoods = zeros((1, len(theta_set)))  # preallocation for speed
for s in range(len(theta_set)):
    theta = theta_set[s]
    # Parametric pdf used in the ML estimation
    if theta <= 0:
        nu = 1
        pdf = 1 / sqrt(theta ** 2)*t.pdf((i_t - theta) / theta, nu)
    else:
        pdf = lognorm.pdf(i_t, (theta - 0.01), scale=exp(theta, ))

    loglikelihoods[0,s] = sum(log(pdf))

# ## Choose theta_ML as the value of theta giving rise to the maximum log-likelihood

# +
mmax, max_index = npmax(loglikelihoods),np.argmax(loglikelihoods)
theta_ML = theta_set[max_index]

vars_to_save = {varname: var for varname, var in locals().items() if isinstance(var,(np.ndarray,np.float,np.int))}
savemat(os.path.join(TEMPORARY_DB,'db_MaximumLikelihood'),vars_to_save)
# -

# ## Figure

# print the LL value for range of parameters
figure()
plot(theta_set, loglikelihoods[0], markersize=15,color='b',marker='.',linestyle='none')
# highlight the maximum LL value
scatter(theta_ML, mmax, s=1000, color='r', marker='.',zorder=0)
legend(['Log-likelihoods','Maximum log-likelihood']);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
