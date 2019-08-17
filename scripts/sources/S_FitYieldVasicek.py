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

# # S_FitYieldVasicek [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_FitYieldVasicek&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerVasicekFit).

# ## Prepare the environment

# +
import os
import os.path as path
import sys
from time import sleep

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import array, zeros, ceil, log
from numpy import min as npmin, max as npmax
np.seterr(divide='ignore',invalid='ignore', all='ignore')

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, subplots, ylabel, \
    xlabel
from tqdm import trange

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict
from intersect_matlab import intersect
from RollPrices2YieldToMat import RollPrices2YieldToMat
from FitVasicek import FitVasicek
from ZCBondPriceVasicek import ZCBondPriceVasicek
# -

# ## Upload the database

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_SwapCurve'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_SwapCurve'), squeeze_me=True)

DF_Rolling = struct_to_dict(db['DF_Rolling'])
# -

# ## Compute the yields to maturity from the database and set the initial values for the parameters

# times to maturity
tau = array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]]).T
# len of the time series for fitting
t_ = 40  # low for speed increase to appreciate the homogeneous behavior of the parameters as risk drivers
# yields from rolling pricing
y,_ = RollPrices2YieldToMat(DF_Rolling.TimeToMat, DF_Rolling.Prices)
# extract the last t_end observations for the selected maturities
_, matIndices,_ = intersect(DF_Rolling.TimeToMat, tau)
y = y[matIndices, - t_:]
# initial values for the parameters
par_start = namedtuple('par','theta0 theta1 theta2 theta3')
par_start.theta0 = 0.01
par_start.theta1 = 0.01
par_start.theta2 = 0.2
par_start.theta3 = 0.01

# ## Fit prices and compute yields

print('Fitting Vasicek model')
sleep(0.5)
# preallocating variables
theta = zeros((4, t_))
exit = zeros((1, t_))
res = zeros((1, t_))
z = zeros((len(tau), t_))
y_Vasicek = zeros((len(tau), t_))
for t in trange(t_):
    if t == 0:
        par = FitVasicek(tau, y[:,[t]], par_start)
    else:
        par = FitVasicek(tau, y[:, [t]], par)

    theta[0, t] = par.theta0
    theta[1, t] = par.theta1
    theta[2, t] = par.theta2
    theta[3, t] = par.theta3
    exit[0,t] = par.exit
    res[0,t] = par.res
    # fitted prices
    z[:, [t]] = ZCBondPriceVasicek(tau, par)
    # from prices to yields
    y_Vasicek[:, [t]] = (1/-tau) * log(z[:, [t]])

# ## Generate figures showing the evolution of the parameters and the comparison between the realized and the fitted yield curve at certain points in time

# +
#Vasicek fitted swap curve
n_fig = 1
# number of figures, representing volatility fitting, to be plotted
if n_fig ==1:
    t_fig = range(1)
else:
    t_fig = range(0, t_-1, ceil(t_ / (n_fig - 1)))

for k in t_fig:
    figure()
    plot(tau, y_Vasicek[:,k], 'b', tau, y[:, k], 'r.')
    plt.axis([min(tau), max(tau),npmin(y_Vasicek[:, k]), npmax(y_Vasicek[:, k])])
    xlabel('Time to Maturity')
    ylabel('Rate')
    legend(['Fit','Rates'])
    plt.grid(True);
    # save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

    # parameters evolution
    cellc = ['m','b','g','r']
    celll = [r'$\theta_{0}$',r'$\theta_{1}$',r'$\theta_{2}$',r'$\theta_{3}$']

    f,ax = subplots(4,1)
    for i in range(4):
        plt.sca(ax[i])
        plot(range(t_), theta[i,:], color = cellc[i])
        ylabel(celll[i])
        plt.axis([1, t_,min(theta[i, :]), max(theta[i, :])])
        plt.xticks([])
        plt.grid(True)
    xlabel('Time')
    plt.tight_layout();
    # save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

