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

# # s_kalman_filter_yield_curve [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_kalman_filter_yield_curve&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerKFplot).

# +
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from arpym.estimation import effective_num_scenarios, exp_decay_fp, fit_state_space
from arpym.pricing import fit_nelson_siegel_yield
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_kalman_filter_yield_curve-parameters)

tau = np.arange(1., 11)  # times to maturity
t_ = 650  # length of the time series
tau_p = 6 * 21  # half-life
par_start = np.array([0.5, 0.5, 0.5, 0.5])  # starting parameters for Nels.-Si.
lb = np.array([-0.5, - 0.5, - 0.5, 0])  # lower bounds for the parameters
ub = np.array([0.5, 0.5, 0.5, 1.5])  # upper bounds for the parameters
k_ = 3  # number of factors

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_kalman_filter_yield_curve-implementation-step00): Load data

path = '../../../databases/global-databases/fixed-income/db_yields/'
df_y = pd.read_csv(path + 'data.csv', index_col=0)

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_kalman_filter_yield_curve-implementation-step01): Select the realized yield for time to maturities tau = 1,2,...,10 years

# +
y = np.array(df_y[tau.astype('str')])  # yields to maturity
if y.shape[0] > t_:
    y = y[-t_:, :]
else:
    t_ = y.shape[0]

# increments
dy = np.diff(y, 1, axis=0)  # t_ennd-1 increments
n_ = dy.shape[1]
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_kalman_filter_yield_curve-implementation-step02): Set flexible probabilities and compute effective number of scenarios

p = exp_decay_fp(dy.shape[0], tau_p)
p = p / np.sum(p)  # flexible probabilities
ens = effective_num_scenarios(p)  # effective number of scenarios

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_kalman_filter_yield_curve-implementation-step03): Estimate the evolution of first two Nelson-Siegel parameters

# Nelson-Siegel fit
theta = np.zeros((t_-1, 4))
theta[0, :] = fit_nelson_siegel_yield(tau, y[0, :], par_start)
for t in range(1, t_-1):
    theta[t, :] = fit_nelson_siegel_yield(tau, y[t, :], theta[t-1, :])

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_kalman_filter_yield_curve-implementation-step04): Estimate evolution of first two hidden factors of Kalman Filter

z_KF, alpha, beta, sig2_U, alpha_z, beta_z, sig2_z = fit_state_space(dy, k_, p)
x_rec = alpha + beta@z_KF[-1, :]  # last recovered increment
y_KF = y[t_ - 1, :] + x_rec  # fitted yield curve (using k_ hidden factors)
cum_z_KF = np.cumsum(z_KF[:, :2], axis=0)

# ## Plots

# +
plt.style.use('arpm')

fig1 = plt.figure()
plt.plot(tau, y_KF, 'b', tau, y[t_-1, :], 'r.')
plt.axis([min(tau), max(tau), np.min(y_KF), np.max(y_KF)])
plt.xlabel('Time to Maturity')
plt.ylabel('Rate')
plt.legend(['Fit', 'Rates'])
plt.grid(True)

add_logo(fig1)
plt.tight_layout()

t_plot = t_ - 1
# colors settings
c0 = [1, 0.4, 0.1]
c2 = [0, 0, 0.4]
# tick and labels for the time axes
dates = np.arange(1., t_)
date_tick = np.arange(10, t_plot, 75)
fig2, ax = plt.subplots(2, 1)

# axes for the first hidden factor and first principal component
plt.sca(ax[0])
plt.ylabel('cumulated $z_1$')
plt.plot(dates, cum_z_KF[:, 0], color=c2, lw=0.5)
plt.xticks(dates[date_tick])
plt.axis([min(dates), max(dates), np.min(cum_z_KF[:, 0]),
          np.max(cum_z_KF[:, 0])])

ax2 = ax[0].twinx()
ax2.grid(False)
plt.ylabel('level')
plt.plot(dates, theta[:, 0], color=c0)
plt.axis([min(dates), max(dates), np.min(theta[:, 0]), np.max(theta[:, 0])])

# axes for the second hidden factor and second principal component
plt.sca(ax[1])
plt.axis([min(dates), max(dates), np.min(cum_z_KF[:, 1]),
          np.max(cum_z_KF[:, 1])])
plt.plot(dates, cum_z_KF[:, 1], color=c2, lw=0.5)
plt.xticks(dates[date_tick])
plt.ylabel('cumulated $z_2$')
ax2 = ax[1].twinx()
ax2.grid(False)
plt.plot(dates, theta[:, 1], color=c0)
plt.axis([min(dates), max(dates), np.min(theta[:, 1]), np.max(theta[:, 1])])
plt.ylabel('slope')
add_logo(fig2, location=1)
plt.tight_layout()
