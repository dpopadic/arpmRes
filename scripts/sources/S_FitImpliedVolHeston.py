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

# # S_FitImpliedVolHeston [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_FitImpliedVolHeston&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerImplVolHest).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

from numpy import arange, zeros, linspace, exp, sqrt, tile, array, newaxis, meshgrid
from numpy import min as npmin, max as npmax
import numpy as np

from scipy.interpolate import griddata
from scipy.stats import norm
from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, xlim, ylim, ylabel, \
    xlabel, title, xticks, yticks, subplot
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict, date_mtop
from FitHeston import FitHeston

from tqdm import trange
from blsimpv import blsimpv
from CallPriceHestonFFT import CallPriceHestonFFT

# parameters
r = 0  # risk free rate
z_0 = [2, sqrt(0.1), 0.5, -0.5, 0.1]  # initial guess for Heston parameters: (kappa,sigma_bar,eta,rho,sigma_0)
t_ = 10  # low for speed increase to appreciate the homogeneous behavior of the parameters as risk drivers
# -

# ## Upload the data from db_ImpliedVol_SPX

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_ImpliedVol_SPX'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_ImpliedVol_SPX'), squeeze_me=True)

db_ImpliedVol_SPX = struct_to_dict(db['db_ImpliedVol_SPX'])
# -

# ## Set the initial values for the Heston parameters and select a time-window

# +
dates = db_ImpliedVol_SPX.Dates  # vector of dates
tau = db_ImpliedVol_SPX.TimeToMaturity  # vector of times to maturity
delta = db_ImpliedVol_SPX.Delta  # vector of deltas
sigma_impl = db_ImpliedVol_SPX.Sigma  # matrix of implied volatilities from market quotes
s = db_ImpliedVol_SPX.Underlying  # vector of underlying prices

# selecting the time window
t_obs = len(dates)  # total number of market observations
index = arange(t_obs - t_ ,t_obs)  # index of observations used
dates_t = dates[index]  # selecting the dates
s = s[index]  # selecting the prices of the underlying
sigma_impl = sigma_impl[:,:, index]  # selecting the implied vol
znm = namedtuple('z', 'kappa sigma_bar eta rho sigma_0')
z = znm(zeros(t_),zeros(t_),zeros(t_),zeros(t_),zeros(t_))
# -

# ## Compute the moneyness and the strike grids for the values of delta moneyness and time to maturity contained in the data db_ImpliedVol_SPX
# ## Calibrate, at each point in time, the parameters to ensure the best match of theoretical values with the real values of implied volatility to this purpose use function FitHeston
# ## Compute the implied volatility surface, at each point in time, corresponding to the Heston parameters previously determined

# +
sigma_heston = zeros((len(tau), len(delta), t_))
for t in trange(t_):
    m = norm.ppf(tile(delta[newaxis,...], (len(tau), 1)))*sigma_impl[:,:,t]-(r + sigma_impl[:,:,t]**2/2)*sqrt(tile(tau[...,newaxis], (1, len(delta))))  # moneyness grid
    k = s[t]*exp(-m * sqrt(tile(tau[...,newaxis], (1, len(delta)))))  # strike grid
    z_0, c_heston = FitHeston(tau, k, sigma_impl[:,:,t], r, s[t], z_0.copy())  # computation of Heston parameters and call option prices
    for i in range(len(tau)):
        for j in range(len(delta)):
            sigma_heston[i,j,t]=blsimpv(c_heston[i,j], s[t], k[i,j], r, tau[i])  # computation of the implied volatility from Heston fitted prices
    z.kappa[t] = z_0[0]
    z.sigma_bar[t] = z_0[1]
    z.eta[t] = z_0[2]
    z.rho[t] = z_0[3]
    z.sigma_0[t] = z_0[4]

kappa = z.kappa
sigma_bar = z.sigma_bar
eta = z.eta
rho = z.rho
sigma_0 = z.sigma_0
sigma_heston = sigma_heston*100  # converts values to percentage
sigma_impl = sigma_impl*100  # converts values to percentage
# -

# ## Generate figures showing the evolution of the parameters and the comparison between the realized and the fitted implied volatility surfaces at some points in time

# +
# axes settings
date_tick = linspace(0,t_ - 1,4, dtype=int)
grid_k = linspace(npmin(kappa),npmax(kappa),3)
grid_sigbar = linspace(npmin(sigma_bar),npmax(sigma_bar),3)
grid_lamda = linspace(npmin(eta),npmax(eta),3)
grid_rho = [-1.1, 0, 1.1]
grid_sigma_0 = linspace(min(sigma_0),max(sigma_0),3)
grid_s = [npmin(s), 0.5*(npmin(s) + npmax(s)), npmax(s)]

# volatility fitting plot
n_fig = 1
# number of figures, representing volatility fitting, to be plotted

if n_fig == 1:
    t_fig = [0]
else:
    t_fig = linspace(0,t_-1,n_fig)

for h in t_fig:
    date_dt = array([date_mtop(i) for i in dates_t])
    myFmt = mdates.DateFormatter('%d-%b-%Y')
    f, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
    ax.view_init(32, -133)
    plt.axis([npmin(delta), npmax(delta), npmin(tau), npmax(tau)])
    xlabel('Delta', labelpad=10)
    xticks(delta[::2])
    yticks(tau[[0, 5, 7, 8, 9]])
    ylabel('Maturity', labelpad=10)
    ax.set_zlabel('$\sigma$')
    ax.set_zticks([10, 20, 30, 40], ['10%', '20%', '30%', '40%'])
    title('HESTON IMPLIED VOLATILITY SURFACE SP500')
    plt.grid(True)
    interpoints = 100
    delta_plot, tau_plot = meshgrid(delta, tau)
    xi = linspace(min(delta),max(delta),interpoints)
    yi = linspace(min(tau),max(tau),interpoints)
    xiplot, yiplot = meshgrid(xi,yi)
    grid_mkt = griddata((delta_plot.flatten(), tau_plot.flatten()), sigma_impl[:,:,h].flatten(), (xiplot.flatten() , yiplot.flatten()), method='cubic').reshape(xiplot.shape)
    grid_heston = griddata((delta_plot.flatten(), tau_plot.flatten()), sigma_heston[:,:,h].flatten(), (xiplot.flatten() , yiplot.flatten()), method='cubic').reshape(xiplot.shape)
    m, n = xiplot.shape
    colors = array([[0, 0, 0.7]]*(m*n)).reshape(m,n,-1)
    # colorsheston = array([[0.9, 0, 0.40]]*(m*n)).reshape(m,n,-1)
    mask = grid_mkt<grid_heston
    colors[mask,:] = [0.9, 0, 0.40]
    ax.plot_surface(xiplot, yiplot, grid_mkt, alpha=.65, facecolors=colors, rstride=1, cstride=1)
    # ax.plot_surface(delta_plot, tau_plot, np.maximum(sigma_impl[:, :, h],sigma_heston[:, :, h]), alpha=.65, facecolors=colors, rstride=1, cstride=1)
    # ax.plot_surface(delta_plot, tau_plot, sigma_heston[:, :, h], alpha=.75, color=[0.9, 0, 0.40])
    ax.plot([0], [0], [0], lw=1, color=[0, 0, 0.7], label='Mkl')
    ax.plot([0], [0], [0], lw=1, color=[0.9, 0, 0.40], label='Heston')
    ax.set_zlim([10, 40])
    ax.legend(loc=2)
    par_string = '\n'.join(
        [date_dt[0].strftime('%d-%b-%y'), '$s_0$: %1.2f' % s[h], '$\kappa$: %1.2f' % kappa[h],
         '$\eta$: %1.2f' % eta[h], r'$\rho$: %1.2f' % rho[h],
         '$\sigma_{0}$ %3.0f' % sqrt(sigma_0[h]), '$\overline{\sigma}$: %1.2f' % sigma_bar[h]])
    ax.text(1.33, 1.7, 10, par_string);
    # save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
    # parameter plotsr
    f, ax = plt.subplots(3, 1)
    plt.sca(ax[0])
    title('EVOLUTION OF THE PARAMETERS')
    plot(date_dt, kappa, color='b', lw=1.5)
    yticks(grid_k)
    ylim([npmin(kappa), npmax(kappa)])
    ylabel('$\kappa$', color='b')
    plt.grid(True)
    xticks([])
    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2_h2 = ax[0].twinx()
    plot(date_dt, sigma_bar, color='r', lw=1.5)
    ylim([min(sigma_bar), max(sigma_bar)])
    yticks(grid_sigbar)
    ax2_h2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ylabel(r'$\bar{\sigma}$', color='r')
    ax[0].set_xlim([min(date_dt), max(date_dt)])
    ax[0].xaxis.set_major_formatter(myFmt)
    plt.sca(ax[1])
    plot(date_dt, eta, color='c', lw=1.5)
    ylim([min(eta), max(eta)])
    xticks([])
    yticks(grid_lamda)
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ylabel('$\eta$', color='c')
    plt.grid(True)
    ax2 = ax[1].twinx()
    ylim([-1.1, 1.1])
    yticks([-1, 0, 1])
    ylabel(r'$\rho$', color='k')
    ax2.plot(date_dt, rho, color='k', lw=1.5)
    ax[1].set_xlim([min(date_dt), max(date_dt)])
    ax[1].xaxis.set_major_formatter(myFmt)
    plt.sca(ax[2])
    plt.axis([min(date_dt), max(date_dt), min(sigma_0), max(sigma_0)])
    xticks(date_dt[date_tick])
    yticks(grid_sigma_0)
    xlab = xlabel('Date')
    plot(date_dt, sigma_0, color='g', lw=1.5)
    ylabel('$\sigma_0$', color='g')
    plt.grid(True)
    ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2_h4 = ax[2].twinx()
    ax2_h4.plot(date_dt, s, color='m', lw=1.5)
    ylim([min(s), max(s)])
    yticks(grid_s)
    ylabel('$s_0$', color='m')
    ax[2].set_xlim([min(date_dt), max(date_dt)])
    ax[2].xaxis.set_major_formatter(myFmt)
    ax2_h4.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    plt.tight_layout();
    # save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

