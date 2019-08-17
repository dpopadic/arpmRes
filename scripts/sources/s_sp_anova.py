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

# # s_sp_anova [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_sp_anova&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_sp_anova).

# +
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from arpym.statistics import meancov_sp
from arpym.tools import histogram_sp
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_sp_anova-parameters):

delta_t = 100  # horizon parameter
sigma_ = None  # clustering threshold

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_sp_anova-implementation-step00): Load data

path = '../../../databases/temporary-databases/'
file = 'db_call_data.csv'
t_end = pd.read_csv(path+file,
                    usecols=['m_'], nrows=1).values[0, 0].astype(int)+1
j_ = pd.read_csv(path+file,  # number of scenarios
                 usecols=['j_'], nrows=1).values[0, 0].astype(int)
data = pd.read_csv(path+file, usecols=['log_sigma_atm', 'log_s'])
log_v_sandp = data.log_s.values.reshape(j_, t_end)
# implied volatility surface at the money
log_sigma_atm = data.log_sigma_atm.values.reshape(j_, t_end)
del data

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_sp_anova-implementation-step01): Compute returns and expected returns at t_hor

# +
# define t_now and t_hor as indexes
t_now = 0
t_hor = t_now + delta_t-1
# extract values of the S&P index at t_now and t_hor
v_sandp_tnow = np.exp(log_v_sandp[0, 0])
v_sandp_thor = np.exp(log_v_sandp[:, delta_t-1])
#extract horizon values of the implied volatility at the money
sigma_atm_thor = np.exp(log_sigma_atm[:, delta_t-1])

# compute returns of the S&P 500 index between t_now and t_hor
r_sandp = (v_sandp_thor/v_sandp_tnow - 1)
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_sp_anova-implementation-step02): If not given determine the optimal clustering

if sigma_ is None:

    def intra_cluster_variance(sigma):
        sigma_atm_thor_id = sigma_atm_thor > sigma
        r_sandp_given_0 = r_sandp[~sigma_atm_thor_id]
        r_sandp_given_1 = r_sandp[sigma_atm_thor_id]
        _, cv_r_sandp_given_0 = meancov_sp(r_sandp_given_0)
        _, cv_r_sandp_given_1 = meancov_sp(r_sandp_given_1)
        p = r_sandp_given_1.shape[0]/j_
        return (1-p)*cv_r_sandp_given_0 + p*cv_r_sandp_given_1

    sigma_ = minimize_scalar(intra_cluster_variance,
                             bounds=(sigma_atm_thor.min(),
                                     sigma_atm_thor.max()),
                             method='bounded').x

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_sp_anova-implementation-step03): Find the best predictor

# +
# scenarios of the conditional
sigma_atm_thor_id = np.abs(sigma_atm_thor) > sigma_
r_sandp_given_0 = r_sandp[~sigma_atm_thor_id]
r_sandp_given_1 = r_sandp[sigma_atm_thor_id]
# conditional expectation
m_x_0, _ = meancov_sp(r_sandp_given_0)
m_x_1, _ = meancov_sp(r_sandp_given_1)

# ANOVA predictor
def chi(z):
    return m_x_0*(z <= sigma_) + m_x_1*(z > sigma_)
# -

# ## Plots:

# +
plt.style.use('arpm')

# marginal distributions
p_0 = r_sandp_given_0.shape[0]/j_
p_1 = r_sandp_given_1.shape[0]/j_
f_r_sandp_0, bin0 = histogram_sp(r_sandp_given_0,
                                 k_=int(np.log(len(r_sandp_given_0))))
f_r_sandp_1, bin1 = histogram_sp(r_sandp_given_1,
                                 k_=int(np.log(len(r_sandp_given_1))))

# colors
teal = [0.2344, 0.582, 0.5664]
light_green_1 = [0.8398, 0.9141, 0.8125]
light_green_2 = [0.4781, 0.6406, 0.4031]
light_grey = [0.7, 0.7, 0.7]
orange = [0.94, 0.35, 0]
markersize = 60
j_plot = 100  # number of plotted simulations
xlim = [-0.1, 1.1]
ylim = [max(bin0[0], bin1[0]), min(bin0[-1], bin1[-1])]
matplotlib.rc('axes', edgecolor='none')

fig = plt.figure()
# plot locations
shift = -0.1
pos1 = [0.346+shift, 0.2589, 0.56888, 0.7111]
pos2 = [0.346+shift, 0.03, 0.56888, 0.1889]
pos3 = [0.157+shift, 0.2589, 0.16, 0.7111]

# top right plot
ax1 = fig.add_axes(pos1)
ax1.set_xlim(xlim)
ax1.set_xticks([0, 1])
ax1.tick_params(axis='both', direction='out', colors='none')
ax1.set_xlabel(r'$1_{\Sigma_{\mathit{ATM}}>\bar{\sigma}}$',
               labelpad=-20, fontsize=20)
ax1.set_ylabel(r'$R^{\mathit{S&P}}$', labelpad=-35,
               fontsize=20)
# lines through means
ax1.plot(xlim, [m_x_0, m_x_0], xlim,
         [m_x_1, m_x_1],
         c=light_green_2, lw=0.5)
# joint
l1 = ax1.scatter(sigma_atm_thor_id[:j_plot], r_sandp[:j_plot],
                 s=markersize*3, edgecolor=light_grey, c=['none'], marker='o')
# conditional expectation
l4 = ax1.scatter(0, m_x_0, marker='x', s=markersize*3, c=[orange], lw=6)
ax1.scatter(1, m_x_1, marker='x', s=markersize*3, c=[orange], lw=3)
ax1.set_title('Analysis of variance',
              fontdict={'fontsize': 20, 'fontweight': 'bold'})

# bottom plot
ax2 = fig.add_axes(pos2, sharex=ax1)
ax2.set_xlim(xlim)
ax2.set_ylim([-0.01, 1.001])
ax2.set_yticks([0, 0.5, 1])
ax2.grid(True, color=light_grey)

l2 = ax2.bar(0, p_0, 0.2, color=light_green_2, align='center')
ax2.bar(1, p_1, 0.2, bottom=p_0, color=light_green_2, align='center')
ax2.plot([0.1, 0.9], [p_0, p_0], c=light_green_2, lw=0.5)

# left plot
ax3 = fig.add_axes(pos3, sharey=ax1)
ax3.set_xlim([0, 1.1*np.max(np.r_[f_r_sandp_0, f_r_sandp_1])])
ax3.set_xticks([])
ax3.invert_xaxis()
# conditional pdf's
ax3.plot(f_r_sandp_0, bin0, c=light_green_2, lw=2)
l3, = ax3.plot(f_r_sandp_1, bin1, c=light_green_2, lw=1)
ax1.set_xlim(xlim)
ax1.set_ylim(ylim)

# legend
fig.legend((l3, l4, l2, l1),
           (r'Conditional $R^{\mathit{S&P}}|1_{\Sigma_{\mathit{ATM}}>\bar{\sigma}}$',
            'Optimal prediction',
            r'Marginal $1_{\Sigma_{\mathit{ATM}}>\bar{\sigma}}$',
            r'Joint $(R^{\mathit{S&P}},1_{\Sigma_{\mathit{ATM}}>\bar{\sigma}})$'),
           loc=(0.42, 0.75), prop={'size': '17', 'weight': 'bold'},
           edgecolor='none', facecolor='none')
add_logo(fig)
