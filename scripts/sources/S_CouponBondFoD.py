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

# # S_CouponBondFoD [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_CouponBondFoD&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-fo-dcoupon-bond).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import array, zeros, where, squeeze, \
    cov, round, log, sqrt, tile, r_
from numpy import sum as npsum, max as npmax
from numpy.linalg import lstsq

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, ylabel, \
    xlabel, title

plt.style.use('seaborn')

from ARPM_utils import save_plot
from HistogramFP import HistogramFP
# -

# ## Run S_PricingCouponBondTaylor that computes the first and second order Taylor approximations of the coupon bond P&L

from S_PricingCouponBondTaylor import *

# ## Decompose the coupon bond P&L into the numerical yield, the effective duration
# ## and the effective convexity contributions using the Factors on Demand framework.

# +
i_ = where(horiz_u == 0.5)[0][0]  # selected horizon (6 months)
delta_t = tile(horiz_u[:i_+1].reshape(1,-1) - horiz_u[0], (j_, 1))  # time changes
y0 = tile(y[:,[-1]], (1, j_))
delta_y = X_u - tile(y0[...,np.newaxis], (1, 1, u_))  # yield changes
delta_y2 = delta_y ** 2
greeks = r_['-1',array([[y_hat]]), dur_hat.T, conv_hat.T]
g_ = greeks.shape[1]
beta = zeros((g_, i_))
MargRiskContr = zeros((g_ + 1, i_+1))
MargRiskContr_dur = zeros((1, i_+1))
MargRiskContr_con = zeros((1, i_+1))

for i in range(1, i_+1):
    X = r_['-1', delta_t[:, [i]], -delta_y[:,:, i].T, 0.5*delta_y2[:,:, i].T]
    Yr = PL_u[:,[i]]-PL_u[0,[i]]
    b = lstsq(X, Yr)[0]
    r = X @ b - Yr
    Z = r_['-1',r, X]
    beta = r_[array([[1]]), b]
    MargRiskContr[:,[i]] = beta * ((cov(Z.T)@beta) / sqrt(beta.T@cov(Z.T)@beta))
    MargRiskContr_dur[0,[i]] = npsum(MargRiskContr[2:10, i])  # effective duration contribution
    MargRiskContr_con[0,[i]] = npsum(MargRiskContr[9:16, i])  # effective convexity contribution
# -

# ## Plot few (say 15) simulated paths of the coupon bond P&L up to the selected horizon (9 months),
# ## along with the mean and the standard deviation of the projected P&L.
# ## Furthermore, show the contributions given by the numerical yield, the effective duration and the effective convexity.

lgrey = [0.8, 0.8, 0.8]  # light grey
dgrey = [0.4, 0.4, 0.4]  # dark grey
j_sel = 15  # selected MC simulations
figure()
# simulated paths
plot(horiz_u[:i_+1], PL_u[:j_sel,:i_+1].T, color=lgrey,zorder=0)
# histogram
option = namedtuple('option', 'n_bins')
option.n_bins = round(10*log(j_))
y_hist, x_hist = HistogramFP(PL_u[:, [i_]].T, pp_, option)
scale = 2*SigmaPL_u[0, i_] / npmax(y_hist)
y_hist = y_hist*scale
shift_y_hist = horiz_u[i_] + y_hist[0]
emp_pdf = plt.barh(x_hist[:-1], shift_y_hist-horiz_u[i_], left=horiz_u[i_], height=x_hist[1]-x_hist[0], facecolor=lgrey, edgecolor= lgrey, lw=2) #empirical pdf
plot(shift_y_hist, x_hist[:-1], color=dgrey)  # border
# # delta, vega and gamma components
a1 = plt.fill_between(horiz_u[:i_+1], MuPL_u[0,:i_+1], MuPL_u[0,:i_+1]+ MargRiskContr[1, :i_+1],color='m')  # yield contribution
a2 = plt.fill_between(horiz_u[:i_+1],MuPL_u[0,:i_+1] + MargRiskContr[1, :i_+1],
                      MuPL_u[0,:i_+1] + MargRiskContr[1, :i_+1] + MargRiskContr_dur[0, :i_+1],color='b')  # effective duration contribution
a3 = plt.fill_between(horiz_u[:i_+1], MuPL_u[0,:i_+1] + MargRiskContr[1, :i_+1] + MargRiskContr_dur[0,:i_+1],
                      MuPL_u[0,:i_+1] + MargRiskContr[1, :i_+1] + MargRiskContr_dur[0,:i_+1]+ MargRiskContr_con[0, :i_+1],color='c')  # effective convexity contribution
# # mean and standard deviation of the coupon bond P&L
l1 = plot(horiz_u[:i_+1], MuPL_u[0,:i_+1], color='g')
l2 = plot(horiz_u[:i_+1], MuPL_u[0,:i_+1] + SigmaPL_u[0,:i_+1], color='r')
plot(horiz_u[:i_+1], MuPL_u[0,:i_+1] - SigmaPL_u[0,:i_+1], color='r')
# [l1 l2 emp_pdf a1 a2 a3]
legend(handles=[l1[0], l2[0], emp_pdf[0], a1, a2, a3],labels=['mean',' + / - st.deviation','horizon pdf','yield','duration','convexity'])
xlabel('time (years)')
ylabel('Coupon bond P&L')
title('Coupon bond P&L (no cash-flow) marginal risk contribution');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

