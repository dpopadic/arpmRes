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

# # S_RobustComparHFPvsHBFP [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_RobustComparHFPvsHBFP&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerHBFPellipsoid).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import arange, array, zeros, cos, sin, where, sqrt, tile, r_, diagflat
from numpy.linalg import eig, solve, norm as linalgnorm

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, legend, xlim, ylim, scatter, ylabel, \
    xlabel

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict
from FPmeancov import FPmeancov
from PlotTwoDimEllipsoid import PlotTwoDimEllipsoid
from Price2AdjustedPrice import Price2AdjustedPrice
from GarchResiduals import GarchResiduals
from BlowSpinFP import BlowSpinFP
from ColorCodedFP import ColorCodedFP
from HighBreakdownFP import HighBreakdownFP
# -

# ## Upload the database

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_Stocks'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_Stocks'), squeeze_me=True)

StocksSPX = struct_to_dict(db['StocksSPX'])
# -

# ## Compute the dividend-adjusted returns of two stocks

# +
n_ = 2
t_ = 400

_, x_1 = Price2AdjustedPrice(StocksSPX.Date.reshape(1,-1), StocksSPX.Prices[[25],:], StocksSPX.Dividends[25])  # Cisco Systems Inc
_, x_2 = Price2AdjustedPrice(StocksSPX.Date.reshape(1,-1), StocksSPX.Prices[[5],:], StocksSPX.Dividends[5])  # General Electric
date = StocksSPX.Date[1:]

x_1 = x_1[[0],-t_:]
x_2 = x_2[[0],-t_:]
date = date[-t_:]
# -

# ## Compute the invariants using GARCH(1,1) fit

epsi = GarchResiduals(r_[x_1,x_2])

# ## Compute the Flexible Probability profiles using Blow-Spin method

b = 1  # number of blows
s = 0  # number of spins
p, _ = BlowSpinFP(epsi, b, s)
q_ = b + s

# ## Compute HFP-mean/cov and HBFP-mean/cov from original data

# +
print('Compute HFP - mean / cov and HBFP - mean / cov from original data')

mu_HFP = zeros((n_,2))
mu_HBFP = zeros((n_,2))
sigma2_HFP = zeros((n_,n_,2))
sigma2_HBFP = zeros((n_,n_,2))
p_HBFP = zeros(2)
v_HBFP = zeros(2)

mu_HFP[:, [0]], sigma2_HFP[:, :, 0] = FPmeancov(epsi, p)  # HFP mean and covariance from original data
mu_HBFP[:, 0], sigma2_HBFP[:, :, 0], p_HBFP[0], v_HBFP[0], _ = HighBreakdownFP(epsi, p.copy(),1)  # HBFP mean and covariance from original data
# -

# ## Detect points outside the HBFP ellipsoid

# +
lev = 1.2
Diag_lambda2, e = eig(sigma2_HBFP[:, :, 0])
y = zeros((n_, t_))
ynorm = zeros((1, t_))

for t in range(t_):
    y[:,t] = solve(e@sqrt(diagflat(Diag_lambda2)),epsi[:,t] - mu_HBFP[:, 0])
    ynorm[0,t] = linalgnorm(y[:,t], 2)

selection = where(ynorm > lev)
# -

# ## Shift points outside the HBFP-ellipsoid and compute HFP-mean/cov and HBFP-mean/cov from perturbed data

# +
print('Computing HFP - mean / cov and HBFP - mean / cov from perturbed data')

alpha = 2.9
gamma = 0.27
omega = 0.7

epsi_HBFP= zeros((4,epsi.shape[1]))
epsi_HBFP[0:2] = epsi.copy()
# point-shifting
angle = omega*alpha
rotation = array([[cos(angle),- sin(angle)], [sin(angle), cos(angle)]])
epsi_tilde = tile(mu_HBFP[:, [0]], (1, t_)) + 1.1*e*sqrt(Diag_lambda2)@rotation*(.8 + .2*cos(gamma*alpha))@y
epsi_HBFP[2:]= epsi.copy()
epsi_HBFP[2:, selection] = epsi_tilde[:, selection]  # perturbed dataset
# computation of HFP-mean/cov and HBFP-mean/cov
[mu_HFP[:, [1]], sigma2_HFP[:, :, 1]] = FPmeancov(epsi_HBFP[2:], p)  # HFP-mean/cov from perturbed dataset
mu_HBFP[:, 1], sigma2_HBFP[:, :, 1], p_HBFP[1], v_HBFP[1], _ = HighBreakdownFP(epsi_HBFP[2:], p.copy(), 1)  # HBFP-mean/cov fro
# -

# ## Generate a static figure highlighting the robustness of the HBFP estimators with respect to the corresponding HFP estimators

# scatter colormap and colors
greyrange = arange(0,0.81,0.01)
[CM, C] = ColorCodedFP(p, None, None , greyrange, 0, 1, [0.6, 0.2])
# Xlim and Ylim settings
x_lim = [min(epsi[0]) - .3, max(epsi[0])+.3]
y_lim = [min(epsi[1]) - .3, max(epsi[1])+.3]
for k in range(2):
    f=figure()
    xlim(x_lim)
    ylim(y_lim)
    ell_HFP=PlotTwoDimEllipsoid(mu_HFP[:,[k]], sigma2_HFP[:,:,k], 1, False, False, 'b', 2)
    ell_HBFP = PlotTwoDimEllipsoid(mu_HBFP[:,[k]], sigma2_HBFP[:,:,k], 1, False, False, 'r', 2)
    shift=scatter(epsi_HBFP[2*k, selection], epsi_HBFP[2*k+1, selection], 15, facecolor='none', edgecolor=[.8, .5, .3],
                  marker='o')
    scatter(epsi_HBFP[2*k], epsi_HBFP[2*k+1], 15, c=C, marker='.',cmap=CM)
    xlabel('$\epsilon_1$')
    ylabel('$\epsilon_2$')
    plt.grid(True)
    plt.gca().set_facecolor('white')
    # ell_HFP, ell_HBFP, shift
    leg = legend(['Historical with Flex.Probs.(HFP): non - robust',
                  'High Breakdown with Flex.Probs.(HBFP): robust',
                  'shifted observations'], loc='best');
    # save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
