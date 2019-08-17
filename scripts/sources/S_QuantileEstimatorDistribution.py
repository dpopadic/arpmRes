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

# # S_QuantileEstimatorDistribution [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_QuantileEstimatorDistribution&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerQuantPdf).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import arange, round
from numpy import max as npmax

from scipy.stats import norm
from scipy.misc import comb

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, scatter, title

plt.style.use('seaborn')

from ARPM_utils import save_plot
# -

# ## Compute the analytical pdf of the HFP-quantile estimator

# +
t_ = 150  # len of the time series

c = 0.35  # confidence level

x = arange(-4,4.01,0.01)
F_x = norm.cdf(x)
f_x = norm.pdf(x)
ct_ = round(c*t_)
# HFP-quantile estimator's pdf
f_qc = ct_*comb(t_, ct_)*(F_x ** (ct_ - 1)) * ((1 - F_x) ** (t_ - ct_)) * f_x
# -

# ## Compute the true value for the HFP-quantile

q_c = norm.ppf(c)

# ## Create figure

# +
blue = [0, .4, 1]
green = [0.1, 0.8, 0]

figure()
HFP = plot(x, f_qc, lw=1.5,color=blue)
TRUE = scatter([q_c, q_c], [0, 0], 40, green, marker='.')
legend(['HFP-quantile pdf','true quantile'])
title('pdf of order statistics')
T = 'time series len =  % 3.0f'%t_
plt.text(3.93, 0.84*npmax(f_qc), T, color='k',horizontalalignment='right')
C = 'confidence level =  % 1.2f' %c
plt.text(3.93, 0.77*npmax(f_qc), C, color='k',horizontalalignment='right');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
