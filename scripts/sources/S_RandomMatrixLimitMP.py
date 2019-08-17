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

# # S_RandomMatrixLimitMP [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_RandomMatrixLimitMP&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=MarchenkoPasturLimit).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import ceil, cov, log, exp, sqrt, histogram
from numpy.linalg import eig
from numpy.random import rand, randn

from scipy.stats import expon, lognorm

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, bar, legend, title

plt.style.use('seaborn')

from ARPM_utils import save_plot
from MarchenkoPastur import MarchenkoPastur

# initialize variables
t_ = 1500
i_ = 300
# -

# ## Generate samples

X_1 = randn(i_, t_)  # ## standard normal
X_2 = expon.rvs(scale=1, size=(i_, t_)) - 1  # exponential
X_3 = (rand(i_, t_) - 0.5)*sqrt(12)  # uniform
X_4 = (lognorm.rvs(1,scale=1, size=(i_, t_)) - exp(0.5)) / sqrt(exp(2) - exp(1))  # log-normal

# ## Compute the covariance matrices

Sigma2_1 = cov(X_1, ddof=1)  # ## standard normal
Sigma2_2 = cov(X_2, ddof=1)  # exponential
Sigma2_3 = cov(X_3, ddof=1)  # uniform
Sigma2_4 = cov(X_4, ddof=1)  # log-normal

# ## Compute the sample eigenvalues and the corresponding normalized histograms

# +
nbins = int(ceil(10*log(i_)))

# standard normal
Lambda2_1,_ = eig(Sigma2_1)
hgram_1, x_1 = histogram(Lambda2_1, nbins)
d = x_1[1] - x_1[0]
hgram_1 = hgram_1 / (d*i_)
# exponential
Lambda2_2,_ = eig(Sigma2_2)
hgram_2, x_2 = histogram(Lambda2_2, nbins)
d = x_2[1] - x_2[0]
hgram_2 = hgram_2 / (d*i_)
# uniform
Lambda2_3,_= eig(Sigma2_3)
hgram_3, x_3 = histogram(Lambda2_3, nbins)
d = x_3[1] - x_3[0]
hgram_3 = hgram_3 / (d*i_)
# log-normal
Lambda2_4,_ = eig(Sigma2_4)
hgram_4, x_4 = histogram(Lambda2_4, nbins)
d = x_4[1] - x_4[0]
hgram_4 = hgram_4 / (d*i_)
# -

# ## Compute the Marchenko-Pastur limit of the empirical eigenvalues' distribution

# +
q = t_ / i_

l_ = 1500  # coarseness
x_mp, y_mp, _ = MarchenkoPastur(q, l_, 1)
# -

# ## Create figures

# +
# standard normal
figure()
bar(x_1[:-1], hgram_1,width=x_1[1]-x_1[0], facecolor= [.7, .7, .7], edgecolor= [.5, .5, .5],label='Sample eigenvalues')
plot(x_mp, y_mp, 'r',lw= 2,label='Marchenko-Pastur limit')
title('Standard Normal variables')
legend();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

# exponential
figure()
bar(x_2[:-1], hgram_2,width=x_2[1]-x_2[0], facecolor= [.7, .7, .7], edgecolor= [.5, .5, .5],label='Sample eigenvalues')
plot(x_mp, y_mp, 'r',lw= 2,label='Marchenko-Pastur limit')
title('Exponential variables')
legend();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

# uniform
figure()
bar(x_3[:-1], hgram_3,width=x_3[1]-x_3[0], facecolor= [.7, .7, .7], edgecolor= [.5, .5, .5],label='Sample eigenvalues')
plot(x_mp, y_mp, 'r',lw= 2,label='Marchenko-Pastur limit')
title('Uniform variables')
legend();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

# log-normal
figure()
bar(x_4[:-1], hgram_4,width=x_4[1]-x_4[0], facecolor= [.7, .7, .7], edgecolor= [.5, .5, .5],label='Sample eigenvalues')
plot(x_mp, y_mp, 'r',lw= 2,label='Marchenko-Pastur limit')
title('Log-normal variables')
legend();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
