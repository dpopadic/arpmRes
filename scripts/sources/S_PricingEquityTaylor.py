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

# # S_PricingEquityTaylor [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_PricingEquityTaylor&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-taylor-equity-pl).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

from numpy import arange, array, ones, zeros, sort, where, diff, linspace, round, log, exp, sqrt
from numpy import sum as npsum, min as npmin, max as npmax

from scipy.stats import norm
from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, xlim, ylabel, \
    xlabel, title, xticks, yticks

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict
from FPmeancov import FPmeancov
from HistogramFP import HistogramFP
from SaddlePointQuadN import SaddlePointQuadN
from SimulateBrownMot import SimulateBrownMot
# -

# ## run S_PricingEquityProfitAndLoss

from S_PricingEquityProfitAndLoss import *

# ## Compute the first order approximation of the equity P&L, which has normal distribution,
# ## and the second order approximation of the equity P&L, that has generalized non-central gamma distribution,
# ## at the selected horizon (120 days). Use function SaddlePointQuadN to compute the cumulative distribution
# ## function of the generalized non-central gamma distribution.

# +
n_ = 500
hor_sel = 120  # selected horizon for the plot (120 days)
i = where(horiz_u == hor_sel)[0][-1]
x_hor = zeros((n_, i+1))
Taylor_first = zeros((n_, i+1))
cdf_QuadN = zeros((n_, i+1))
Taylor_second = zeros((n_, i+1))

x_hor[:,i] = linspace(Mu_PL[0, i] - 10*Sigma_PL[0, i], Mu_PL[0, i] + 10 *Sigma_PL[0, i], n_)
# first order approximation (normal pdf)
Taylor_first[:,i] = norm.pdf(x_hor[:,i], exp(x[0,-1])*mu*horiz_u[i + 1],exp(x[0,-1])*sig*sqrt(horiz_u[i + 1]))
# second order approximation (QuadN pdf)
b, c, mu2, sigma2 = array([[exp(x[0,-1])]]), array([[exp(x[0,-1])*0.5]]), mu*horiz_u[i + 1], sig**2*horiz_u[i + 1]
_, Taylor_second[:,i] = SaddlePointQuadN(x_hor[:,[i]].T, 0, b, c, mu2, sigma2)  # QuadN cumulative density function
# Taylor_second(:,i) = diff(cdf_QuadN(:,i))/diff(x_hor((:,i)))
# -

# ## Plot a few (say 15) simulated paths of the equity P&L up to the selected horizon (120 days),
# ## along with the first order approximation, the second order approximation and the analytical
# ## distribution of the equity P&L. Furthermore, show the mean and the standard deviation of
# ## the analytical distribution.

# +
lgrey = [0.8, 0.8, 0.8]  # light grey
dgrey = [0.4, 0.4, 0.4]  # dark grey
lblue = [0.27, 0.4, 0.9]  # light blu
orange = [0.94, 0.35, 0]  # orange
j_sel = 15  # selected MC simulations

figure()
# simulated path, mean and standard deviation
plot(horiz_u[:i], PL[:j_sel,:i].T, color=lgrey)
plt.xticks(arange(0,t_end+20,20))
xlim([npmin(horiz_u), npmax(horiz_u)+1])
l1 = plot(horiz_u[:i], Mu_PL[0,:i], color='g', label='mean')
l2 = plot(horiz_u[:i], Mu_PL[0,:i] + Sigma_PL[0,:i], color='r', label=' + / - st.deviation')
plot(horiz_u[:i], Mu_PL[0,:i] - Sigma_PL[0,:i], color='r')

# analytical pdf
flex_probs_scenarios = ones((1, j_)) / j_
option = namedtuple('option','n_bins')
option = namedtuple('option', 'n_bins')
option.n_bins = round(10 * log(j_))
y_hist, x_hist = HistogramFP(PL[:,[i]].T, flex_probs_scenarios, option)
scale = 0.15 * Sigma_PL[0, i] / npmax(y_hist)
y_hist = y_hist * scale
shift_y_hist = horiz_u[i] + y_hist
emp_pdf = plt.barh(x_hist[:-1], shift_y_hist[0]-npmin(shift_y_hist[0]), height=x_hist[1]-x_hist[0],
         left=npmin(shift_y_hist[0]), facecolor=lgrey, edgecolor=lgrey,label='horizon pdf')  # empirical pdf
plot(shift_y_hist[0], x_hist[:-1], color=dgrey)  # border

# first order approximation
Taylor_first[:,i] = Taylor_first[:,i]*scale
shift_T_first = horiz_u[i] + Taylor_first[:,i]
l3 = plot(shift_T_first, x_hor[:,i], color=orange, label='first order approx')

# second order approximation
Taylor_second[:,i] = Taylor_second[:,i]*scale
shift_T_second = horiz_u[i] + Taylor_second[:,i]
l4 = plot(shift_T_second, x_hor[:,i], color=lblue, label='second order approx')

legend()
xlabel('time (days)')
ylabel('P&L')
title('P&L equity Taylor approximation');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
