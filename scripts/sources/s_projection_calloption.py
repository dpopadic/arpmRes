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

# # s_projection_calloption [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_projection_calloption&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-shadowrateproj-mc).

# +
import numpy as np
import pandas as pd
from scipy.stats import t as tstu
import matplotlib.pyplot as plt

from arpym.statistics import simulate_t, quantile_sp, meancov_sp
from arpym.tools import add_logo, histogram_sp
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_projection_calloption-parameters)

m_ = 120  # number of monitoring times
j_ = 1000  # number of Monte Carlo scenarios

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_projection_calloption-implementation-step00): Import data

# +
path = '../../../databases/temporary-databases/'

# upload GARCH parameters
db_garch = pd.read_csv(path+'db_calloption_garch.csv')
a_garch = db_garch['a'][0]
b_garch = db_garch['b'][0]
c_garch = db_garch['c'][0]
mu_garch = db_garch['mu'][0]
sig2_garch_prev = db_garch['sig2prev'][0]
x_tnow_s = db_garch['x_tnow'][0]
x_tnowm1_s = db_garch['x_tnow-1'][0]

# VAR(1) parameter b
db_var1 = pd.read_csv(path+'db_calloption_var1.csv')
x_tnow_sigma = db_var1.loc[:, db_var1.columns == 'x_tnow'].values.reshape(-1)
b_hat = db_var1.loc[:, db_var1.columns != 'x_tnow'].values

# realized invariants
db_epsi_var1 = pd.read_csv(path+'db_calloption_epsi_var1.csv', index_col=0,
                           parse_dates=True)
epsi_var1 = db_epsi_var1.values
db_epsi_garch = pd.read_csv(path+'db_calloption_epsi_garch.csv', index_col=0,
                            parse_dates=True)
epsi_garch = db_epsi_garch.values
epsi = np.c_[epsi_garch, epsi_var1]
t_, i_ = epsi.shape
t_ = t_ + 1

# flexible probabilities and parameters of t copula
db_estimation = pd.read_csv(path+'db_calloption_estimation.csv')
p = db_estimation['p'].values
nu = db_estimation['nu'][0]
rho2 = db_estimation.loc[:i_-1, np.logical_and(db_estimation.columns != 'nu',
                                               db_estimation.columns != 'p')]
rho2 = rho2.values
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_projection_calloption-implementation-step01): Monte Carlo scenarios for the invariants

epsi_proj = np.zeros((j_, m_, i_))
for m in range(m_):
    # simulate t-copula
    epsi_tilde_proj = simulate_t(np.zeros(i_), rho2, nu, j_)

    # copula scenarios
    for i in range(i_):
        # cdf of marginal distribution
        u_proj = tstu.cdf(epsi_tilde_proj[:, i], nu)
        # quantiles of marginals
        epsi_proj[:, m, i] = \
            quantile_sp(u_proj, epsi[:, i], p).squeeze()

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_projection_calloption-implementation-step02): Compute paths of the risk drivers

# +
x_tnow_thor = np.zeros((j_, m_+1, i_))
x_tnow_thor[:, 0, :] = np.r_[x_tnow_s, x_tnow_sigma]
dx_proj = np.zeros(j_)
dx_proj_prev = np.zeros(j_)
dx_prev = x_tnow_s - x_tnowm1_s

for m in range(m_):
    # GARCH(1,1) projection
    sig2_garch = c_garch + b_garch*sig2_garch_prev + \
                 a_garch*(dx_proj_prev-mu_garch)**2
    dx_proj = mu_garch + np.sqrt(sig2_garch)*epsi_proj[:, m, 0]
    x_tnow_thor[:, m+1, 0] = x_tnow_thor[:, m, 0] + dx_proj
    dx_proj_prev = dx_proj
    sig2_garch_prev = sig2_garch
    # VAR(1) projection
    x_tnow_thor[:, m+1, 1:] = \
        x_tnow_thor[:, m, 1:]@b_hat.T+epsi_proj[:, m, 1:]
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_projection_calloption-implementation-step03): Save databases

# +
out1 = pd.DataFrame({'log_underlying':
                     x_tnow_thor[:, :, 0].reshape(j_*(m_+1))})
out = pd.DataFrame({db_epsi_var1.columns[i-1]:
                    x_tnow_thor[:, :, i].reshape((j_*(m_+1),))
                    for i in range(1, i_)})
out = pd.concat([out1, out], axis=1)

out.to_csv('../../../databases/temporary-databases/db_calloption_proj.csv',
           columns=np.append('log_underlying', db_epsi_var1.columns.values))
del out

# monitoring times
t_now = db_epsi_var1.index[-1]
t_now = np.datetime64(t_now, 'D')
t_m = np.busday_offset(t_now,
                       np.arange(m_+1), roll='forward')
output = {'dates': pd.Series(t_m)}

out = pd.DataFrame(output)
out.to_csv('../../../databases/temporary-databases' +
           '/db_calloption_proj_dates.csv', index=None)
del out
# -

# ## Plots

# +
plt.style.use('arpm')
lgrey = [0.8, 0.8, 0.8]  # light grey
dgrey = [0.4, 0.4, 0.4]  # dark grey

# plot that corresponds to step 4
num_plot = min(j_, 20)

i = 1  # log underlying

mu_thor = np.zeros(m_ + 1)
sig_thor = np.zeros(m_ + 1)
for m in range(0, m_ + 1):
    mu_thor[m], sig2_thor = meancov_sp(x_tnow_thor[:, m, i].reshape(-1, 1))
    sig_thor[m] = np.sqrt(sig2_thor)
fig = plt.figure()
for j in range(num_plot):
    plt.plot(np.arange(0, m_+1), x_tnow_thor[j, :, i], lw=1, color=lgrey)

f, xp = histogram_sp(x_tnow_thor[:, -1, i], k_=20*np.log(j_))
rescale_f = f*5
plt.barh(xp, rescale_f, height=xp[1]-xp[0], left=m_, facecolor=lgrey,
         edgecolor=lgrey)
plt.plot(rescale_f+m_, xp, color=dgrey, lw=1)
# mean plot
p_mu = plt.plot(np.arange(0, m_+1), mu_thor, color='g', label='expectation',
                lw=1)
p_red_1 = plt.plot(np.arange(0, m_+1), mu_thor + 2 * sig_thor,
                   label='+ / - 2 st.deviation', color='r', lw=1)
p_red_2 = plt.plot(np.arange(0, m_+1), mu_thor - 2 * sig_thor, color='r', lw=1)
plt.legend()
plt.xlabel('days')
title = "Log-underlying"
plt.title(title)
add_logo(fig)
fig.tight_layout()

i = 7  # log implied volatility for m=0.05 and tau=0.5
mu_thor = np.zeros(m_ + 1)
sig_thor = np.zeros(m_ + 1)
for m in range(0, m_ + 1):
    mu_thor[m], sig2_thor = meancov_sp(x_tnow_thor[:, m, i].reshape(-1, 1))
    sig_thor[m] = np.sqrt(sig2_thor)
fig1 = plt.figure()
for j in range(num_plot):
    plt.plot(np.arange(0, m_+1), x_tnow_thor[j, :, i], lw=1, color=lgrey)

f, xp = histogram_sp(x_tnow_thor[:, -1, i], k_=20*np.log(j_))
rescale_f = f*10
plt.barh(xp, rescale_f, height=xp[1]-xp[0], left=m_, facecolor=lgrey,
         edgecolor=lgrey)
plt.plot(rescale_f+m_, xp, color=dgrey, lw=1)
p_mu = plt.plot(np.arange(0, m_+1), mu_thor, color='g', label='expectation',
                lw=1)
p_red_1 = plt.plot(np.arange(0, m_+1), mu_thor + 2 * sig_thor,
                   label='+ / - 2 st.deviation', color='r', lw=1)
p_red_2 = plt.plot(np.arange(0, m_+1), mu_thor - 2 * sig_thor,
                   color='r', lw=1)
plt.legend()
plt.xlabel('days')
title = "Point (m=0.05, tau=0.5) on log-implied volatility surface"
plt.title(title)
add_logo(fig1)
fig1.tight_layout()
