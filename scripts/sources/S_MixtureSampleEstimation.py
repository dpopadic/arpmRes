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

# # S_MixtureSampleEstimation [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_MixtureSampleEstimation&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-4-ex-mixture-sim).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import sort, argsort
from numpy import min as npmin, max as npmax
from numpy.random import rand

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot

plt.style.use('seaborn')

from ARPM_utils import save_plot
from QuantileMixture import QuantileMixture
# -

# ## Initialize the parameters

# +
alpha = 0.8  # mixture parameter
mu_Y = 0.1  # location parameter of the normal distribution
sigma_Y = 0.2  # dispersion parameter of the normal distribution
mu_Z = 0  # location parameter of the log-normal distribution
sigma_Z = 0.15  # dispersion parameter of the log-normal distribution

t_ = 52  # len of the sample
# -

# ## Generate the sample of the mixture distribution

# +
p = rand(1, t_)  # t_end realization from a uniform distribution Unif([01])
q = QuantileMixture(p, alpha, mu_Y, sigma_Y, mu_Z, sigma_Z)  # quantiles corresponding to the probability levels p

q_sort, index = sort(q), argsort(q)
p_sort = p[0,index]
# -

# ## Generate figure

f = figure()
p2 = plot(q_sort[0], p_sort[0],'b', marker='*')
plt.axis([npmin(q_sort) - 0.01, npmax(q_sort) + 0.01, npmin(p_sort) - 0.01, npmax(p_sort) + 0.01]);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
