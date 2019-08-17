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

# # S_PtfCompReturn [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_PtfCompReturn&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-comp-ret-ptf).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

from numpy import arange, array, ones, std, round, mean, log, exp, tile, sum as npsum

from scipy.stats import norm

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, bar, legend, xlim, title, yticks

plt.style.use('seaborn')

from ARPM_utils import save_plot
from HistogramFP import HistogramFP
from NormalScenarios import NormalScenarios

# parameters
j_ = 10000  # number of simulations
n_ = 2  # number of instruments in the portfolio
mu = array([[0.01], [0.08]])  # mean of the normal distribution of the instruments compounded returns
sigma2 = array([[0.03, - 0.057], [- 0.057, 0.12]])  # variance of the normal distribution of the instruments compounded returns
w = array([[0.5], [0.5]])  # portfolio weights
# -

# ## Generate j_=10000 normal simulations of the instruments compounded returns
# ## by using function NormalScenarios

Instr_comp_ret = NormalScenarios(mu, sigma2, j_)[0]

# ## Compute the portfolio compounded returns

r = exp(Instr_comp_ret) - 1
r_w = npsum(tile(w, (1, j_)) * r,axis=0,keepdims=True)
ptf_comp_ret = log(1 + r_w)

# ## Compute the normalized empirical histogram stemming from the simulations using function HistogramFP

p = ones((1, j_)) / j_
option = namedtuple('option', 'n_bins')
option.n_bins = round(10*log(j_))
nx, cx = HistogramFP(ptf_comp_ret, p, option)

# ## Plot the histogram of the compounded returns of the portfolio together with the normal fit.

# +
orange = [0.94, 0.3, 0]
blue = [.2, .2, .7]
colhist = [.8, .8, .8]

f = figure()
h = bar(cx[:-1], nx[0],width=cx[1]-cx[0], facecolor= colhist, edgecolor= 'k', label='Port. compounded ret. distr.')

mr = mean(ptf_comp_ret)
sr = std(ptf_comp_ret)
x = arange(min(-3*sr + mr, cx[0] - 0.1), max(3*sr + mr, cx[-1] + 0.1),0.1*sr)
y = norm.pdf(x, mr, sr)
fit = plot(x, y,color= orange,label='Normal fit')
xlim([-0.25, 0.75])
yticks([])

tit = title('Distribution of portfolio compounded returns')
leg = legend();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
