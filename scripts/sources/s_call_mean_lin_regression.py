#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # s_call_mean_lin_regression [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_call_mean_lin_regression&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_call_mean_lin_regression).

# +
import random
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from arpym.pricing import bsm_function, implvol_delta2m_moneyness
from arpym.statistics import meancov_sp
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_call_mean_lin_regression-parameters)

delta_t = 100  # horizon parameter

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_call_mean_lin_regression-implementation-step00): Load data

# +
path = '../../../databases/temporary-databases/'
file = 'db_call_data.csv'
k_strk = pd.read_csv(path+file,  # strike of the underlying
                       usecols=['k_strike'], nrows=1).values[0, 0]
t_end = pd.read_csv(path+file,
                    usecols=['m_'], nrows=1).values[0, 0].astype(int)+1
j_ = pd.read_csv(path+file,  # number of scenarios
                 usecols=['j_'], nrows=1).values[0, 0].astype(int)
data = pd.read_csv(path+file, usecols=['v_call_thor', 'log_s'])
v_call_kstrk_tend = data.v_call_thor.values.reshape(j_, t_end)
log_v_sandp = data.log_s.values.reshape(j_, t_end)
y = pd.read_csv(path+file, usecols=['y_rf'],
                         nrows=1).values[0, 0]

# upload from the database db_implvol_optionSPX
path = '../../../databases/global-databases/derivatives/db_implvol_optionSPX/'
params = pd.read_csv(path+'params.csv')
delta_grid = params.loc[:, 'delta'].dropna().values
tau_grid = params.loc[:, 'time2expiry'].dropna().values
sigma_delta_tnow = pd.read_csv(path+'data.csv', skiprows=712).values[0, 2:]\
        .reshape(delta_grid.shape[0], tau_grid.shape[0]).astype(np.float)
del data
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_call_mean_lin_regression-implementation-step01): Compute returns

# +
# define t_now and t_hor as indexes
t_now = 0
t_hor = t_now + delta_t-1
# extract values of the call option and the underlying at t_now and t_hor
v_call_tnow_kstrk_tend = v_call_kstrk_tend[0, 0]
v_call_thor_kstrk_tend = v_call_kstrk_tend[:, delta_t-1] # scenarios at the horizon of the call option
v_sandp_tnow = np.exp(log_v_sandp[0, 0])
v_sandp_thor = np.exp(log_v_sandp[:, delta_t-1])  # scenarios at the horizon S&P 500 index

# compute returns of the call option and the underlying between t_now and t_hor
r_call = (v_call_thor_kstrk_tend/v_call_tnow_kstrk_tend - 1)
r_sandp = (v_sandp_thor/v_sandp_tnow - 1)
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_call_mean_lin_regression-implementation-step02): Find the best linear predictor

# +
# expectation and covariance of the joint returns
e_r_call_r_sandp, cov_r_call_r_sandp = meancov_sp(np.c_[r_call, r_sandp])
# parameters of the linear mean regression predictor
beta = cov_r_call_r_sandp[0, 1]/cov_r_call_r_sandp[1, 1]
alpha = e_r_call_r_sandp[0]-beta*e_r_call_r_sandp[1]

# linear mean regression predictor
def chi_alpha_beta(z):
    return alpha + beta*np.array(z)
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_call_mean_lin_regression-implementation-step03): Evaluate prediction and residuals

r_bar_call = chi_alpha_beta(r_sandp)  # prediction
u = r_call-r_bar_call  # residuals

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_call_mean_lin_regression-implementation-step04): Compute payoff at t_end and current value of the call option as a function of the underlying S&P500

# +
i_ = 50
# grid of the underlying values
v_sandp_grid = np.linspace(v_sandp_thor.min(), v_sandp_thor.max(), i_)
# time to expiry (in years)
tau = (t_end-t_now)/252
# compute induced m-moneyness
m_sandp = np.log(v_sandp_grid/k_strk)/np.sqrt(tau)

# transition from delta-moneyness to m-moneyness parametrization
sigma_tnow_m_tau, m_grid = implvol_delta2m_moneyness(
    np.array([sigma_delta_tnow.T]), tau_grid, delta_grid, y, tau, 13)
# interpolate impied volatility surface
sigma = interpolate.interp2d(m_grid, tau_grid,
                             sigma_tnow_m_tau.reshape(tau_grid.shape[0], m_grid.shape[0]))
sigma_tnow_m_sandp_tau = sigma(m_sandp, tau)

#current Black-Scholes values of call option as the function of the underlying
v_call_bsm_tnow_kstrk_tend = bsm_function(v_sandp_grid, y,
                                          sigma_tnow_m_sandp_tau, m_sandp, tau)
# payoff of call option as the function of the underlying values
v_call_tend_kstrk_tend = np.maximum(v_sandp_grid-k_strk, 0)
# returns
r_call_bsm_tnow = v_call_bsm_tnow_kstrk_tend/v_call_tnow_kstrk_tend - 1
r_call_tend = v_call_tend_kstrk_tend/v_call_tnow_kstrk_tend - 1
# -

# ## Plots:

# +
# colors
teal = [0.2344, 0.582, 0.5664]
light_green_1 = [0.8398, 0.9141, 0.8125]
light_green_2 = [0.4781, 0.6406, 0.4031]
light_grey = [0.6, 0.6, 0.6]
orange = [0.94, 0.35, 0]
colf = [0, 0.5412, 0.9020]
markersize = 6
# number of plotted simulations
j_plot = random.sample(range(j_), min(j_, 2000))

plt.style.use('arpm')

fig = plt.figure()

k = 0
k1 = 24

ratio = v_sandp_tnow/v_call_tnow_kstrk_tend
ylim = [-1.3, 4]
xstart = -0.3
xlim = [xstart, (ylim[1]-ylim[0])/ratio+xstart]
ax1 = plt.subplot2grid((8, 10), (0, 1), colspan=5, rowspan=5)
ax1.tick_params(axis='x', which='major', pad=-15, direction='out')
ax1.tick_params(axis='y', which='major', pad=-20, direction='out')
ax1.set_xlabel(r'$R^{\mathit{S&P}}$', fontdict={'size': 16}, labelpad=-30)
ax1.set_ylabel(r'$R^{\mathit{call}}$', fontdict={'size': 16}, labelpad=-35)
ax1.scatter(r_sandp[j_plot], r_call[j_plot], s=markersize,
            c=[light_grey])
l5, = ax1.plot(xlim, chi_alpha_beta(xlim), c=orange, lw=1.5)
r_sandp_grid = v_sandp_grid/v_sandp_tnow - 1
l6, = ax1.plot(r_sandp_grid, r_call_bsm_tnow, c='k', lw=1)
l7, = ax1.plot(r_sandp_grid, r_call_tend, '--', c='k', lw=1)
l8, = ax1.plot(e_r_call_r_sandp[1], e_r_call_r_sandp[0], 'o', color=orange)
l9, = ax1.plot(0, 0, 'o', color='k')
ax1.set_title('Linear mean regression',
              fontdict={'fontsize': 20, 'fontweight': 'bold'})
ax2 = plt.subplot2grid((8, 10), (0, 0), colspan=1, rowspan=5, sharey=ax1)
ax2.invert_xaxis()
ax2.hist(r_call, bins='auto', density=True, facecolor=teal, ec=teal,
         orientation='horizontal')
ax2.tick_params(axis='both', colors='none')
ax3 = plt.subplot2grid((8, 10), (5, 1), colspan=5, rowspan=1, sharex=ax1)
ax3.tick_params(axis='both', colors='none')
ax3.invert_yaxis()
ax3.hist(r_sandp, bins='auto', density=True, facecolor=light_green_2,
         ec=light_green_2)
ax1.set_ylim(ylim)
ax1.set_xlim(xlim)

ax4 = plt.subplot2grid((48, 60), (k, 40), colspan=20, rowspan=20)
ax4.tick_params(axis='x', which='major', pad=-16)
ax4.tick_params(axis='y', which='major', pad=-17)
ax4.set_xlabel(r'$\chi_{\alpha,\beta}({R}^{\mathit{S&P}})$',
               fontdict={'size': 16}, labelpad=-32)
ax4.set_ylabel(r'$R^{\mathit{call}}$', fontdict={'size': 16}, labelpad=-33)
ax4.scatter(r_bar_call[j_plot], r_call[j_plot],
            s=markersize, c=[light_grey])
ax6 = plt.subplot2grid((48, 60), (k+20, 40), colspan=20, rowspan=4, sharex=ax4)
ax6.tick_params(axis='both', colors='none')
aaa = ax6.hist(r_bar_call, bins='auto', density=True,
               facecolor=light_green_1, ec=light_green_1)
val, edg = aaa[0], aaa[1]
cent = edg[:-1]+0.5*(edg[1]-edg[0])
ax6.invert_yaxis()
ax5 = plt.subplot2grid((48, 60), (k, 36), colspan=4, rowspan=20, sharey=ax4)
ax5.tick_params(axis='both', colors='none')
ax5.invert_xaxis()
ax5.hist(r_call, bins='auto', density=True, facecolor=teal, ec=teal,
         orientation='horizontal')
ax5.plot(val, cent, color=light_green_1, lw=2)
ax4.set_xlim(ylim)
ax4.set_ylim(ylim)

ax7 = plt.subplot2grid((48, 60), (k1, 40), colspan=20, rowspan=20)
ax7.tick_params(axis='x', which='major', pad=-16)
ax7.tick_params(axis='y', which='major', pad=-27)
ax7.set_xlabel(r'$\chi_{\alpha,\beta}({R}^{\mathit{S&P}})$',
               fontdict={'size': 16}, labelpad=-32)
ax7.set_ylabel('$U$', fontdict={'size': 16}, labelpad=-40)
ax7.scatter(r_bar_call[j_plot], u[j_plot], s=markersize, c=[light_grey])
ax8 = plt.subplot2grid((48, 60), (k1, 36), colspan=4, rowspan=20, sharey=ax7)
ax8.tick_params(axis='both', colors='none')
ax8.tick_params(axis='both', colors='none')
ax8.invert_xaxis()
ax8.hist(u, bins='auto', density=True, facecolor=colf, ec=colf,
         orientation='horizontal')
ax9 = plt.subplot2grid((48, 60), (k1+20, 40), colspan=20, rowspan=4,
                       sharex=ax7)
ax9.tick_params(axis='both', colors='none')
ax9.invert_yaxis()
ax9.hist(r_bar_call, bins='auto', density=True, facecolor=light_green_1,
         ec=light_green_1)
ax7.set_xlim(ylim)
ulim = max(abs(np.percentile(u, 1)), abs(np.percentile(u, 99)))
ax7.set_ylim([-ulim, ulim])

l1 = Rectangle((0, 0), 1, 1, color=light_green_2, ec='none')
l2 = Rectangle((0, 0), 1, 1, color=teal, ec='none')
l3 = Rectangle((0, 0), 1, 1, color=light_green_1, ec='none')
l4 = Rectangle((0, 0), 1, 1, color=colf, ec='none')
fig.legend((l1, l2, l3, l4, l5, l8, l6, l9, l7),
           ('Input', 'Output', 'Predictor', 'Residual', 'LS lin. approx.',
            'Expected returns', 'BMS value', 'Current value', 'Payoff'),
           loc=(0.06, 0.06), prop={'size': '17', 'weight': 'bold'},
           facecolor='none', edgecolor='none', ncol=2, columnspacing=0,
           handletextpad=0.5)
add_logo(fig, axis=ax1, location=1, size_frac_x=1/12)
