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

# # S_CallOptionFoD [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_CallOptionFoD&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-fo-dcall-option).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

from numpy import zeros, where, cov, round, log, sqrt, r_, fliplr, linalg, array

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, ylabel, \
    xlabel, title, fill

plt.style.use('seaborn')

from ARPM_utils import save_plot
from HistogramFP import HistogramFP
# -

# ## run S_PricingCallOptionTaylor which computes the Taylor approximation of the call P&L

from S_PricingCallOptionTaylor import *

# ## Decompose the call option P&L into theta, delta, vega, rho, gamma, vanna
# ## and volga contributions using the Factors on Demand framework.

# +
greeks = [theta_eff, delta_eff, vega_eff, rho_eff, gamma_eff, vanna_eff, volga_eff]
g_ = len(greeks)
beta = zeros((g_, u_ - 1))
MargRiskContr = zeros((g_ + 1, u_ - 1))
i_ = where(horiz_u == 126)[0][0]  # selected horizon (6 months)

for i in range(1,i_+1):
    X = r_['-1',np.array([delta_t]).T, np.array([delta_s]).T, np.array([delta_sig]).T, np.array([delta_y]).T, np.array([0.5*delta_s**2]).T,np.array([delta_sig*delta_s]).T, np.array([0.5*delta_sig**2]).T]
    Yr = PLC_u[:,[i]]
    b = linalg.lstsq(X,Yr)[0]
    r = X@b-Yr
    Z = r_['-1',r, X]
    beta = r_[array([[1]]), b]
    MargRiskContr[:,[i]] = beta * ((cov(Z.T)@beta) / sqrt(beta.T@cov(Z.T)@beta))
# -

# ## Plot a few simulated paths of the call option P&L up to the selected horizon (6 months),
# ## along with the mean and the standard deviation of the projected P&L.
# ## Furthermore, show the contributions given by delta, vega and gamma components, along with the residual.

# +
lgrey = [0.8, 0.8, 0.8]  # light grey
dgrey = [0.4, 0.4, 0.4]  # dark grey
j_sel = 15  # selected MC simulations

figure()
# simulated paths
plot(horiz_u[:i_+1], PLC_u[:j_sel, :i_+1].T, color=lgrey,zorder=0)
# histogram
option = namedtuple('option', 'n_bins')
option.n_bins = round(10*log(j_))
y_hist, x_hist = HistogramFP(PLC_u[:, [i_]].T, pp_, option)
scale = 0.5*PLSigmaC_u[0, i_] / npmax(y_hist)
y_hist = y_hist*scale
shift_y_hist = horiz_u[i_] + y_hist
emp_pdf = plt.barh(x_hist[:-1], shift_y_hist[0]-horiz_u[i_], left=horiz_u[i_], height=x_hist[1]-x_hist[0],
                   facecolor=lgrey, edgecolor= lgrey, lw=2,zorder=20) #empirical pdf
# delta, vega and gamma components
a1 = plt.fill_between(horiz_u[:i_+1],PLMuC_u[0,:i_+1], PLMuC_u[0,:i_+1] + MargRiskContr[2, :i_+1], color='b')  # delta contribution
a2 = plt.fill_between(horiz_u[:i_+1],PLMuC_u[0,:i_+1] + MargRiskContr[2, :i_+1],
                      PLMuC_u[0,:i_+1]+ MargRiskContr[2, :i_+1] + MargRiskContr[3, :i_+1], color='m')  # vega contribution
a3 = plt.fill_between(horiz_u[:i_+1],PLMuC_u[0,:i_+1] + MargRiskContr[2, :i_+1] + MargRiskContr[3, :i_+1],
                      PLMuC_u[0,:i_+1] + MargRiskContr[2, :i_+1] + MargRiskContr[3, :i_+1] + MargRiskContr[5,:i_+1],color='c')  # gamma contribution
# mean and standard deviation of the call option P&L
l1 = plot(horiz_u[:i_+1], PLMuC_u[0, :i_+1], color='g')
l2 = plot(horiz_u[:i_+1], PLMuC_u[0, :i_+1] + PLSigmaC_u[0, :i_+1], color='r')
plot(horiz_u[:i_+1], PLMuC_u[0, :i_+1] - PLSigmaC_u[0, :i_+1], color='r')
legend(handles=[l1[0], l2[0], emp_pdf[0], a1, a2, a3],labels=['mean',' + / - st.deviation','horizon pdf','delta','vega','gamma'])
xlabel('time (days)')
ylabel('Call option P&L')
title('Call option P&L marginal risk contribution');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
