#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # s_checklist_executive_summary [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_checklist_executive_summary&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-executive-summary).

# +
import numpy as np
import pandas as pd

from arpym.statistics import meancov_sp
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_executive_summary-parameters)

deltat = 5.0  # horizon span (business bdays)
h = np.array([2*10**6, 8*10**5])  # vector of holdings
v_risky = 40*10**6  # budget of dollars at risk
t_first = pd.to_datetime('16-03-2012')  # first considered date
t_now = pd.to_datetime('30-03-2012')  # last considered date

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_executive_summary-implementation-step00): Load data

path = '../../../databases/global-databases/equities/'
data = pd.read_csv(path+'db_stocks_SP500/SPX.csv',
                   parse_dates=True, index_col=0)
v_sandp = data[(data.index >= t_first) &
               (data.index <= t_now)].values.reshape(-1)
data1 = pd.read_csv(path+'db_stocks_SP500/db_stocks_sp.csv',
                   parse_dates=True, index_col=0, header=1,
                   usecols=['name', 'CVC', 'AON'])
v_stocks = data1[(data1.index >= t_first) &
                 (data1.index <= t_now)].values
v_n_t = np.c_[v_stocks, v_sandp][:, [1, 0, 2]]
del data, data1

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_executive_summary-implementation-step01): Risk drivers identification

# Compute the time series of the log values
x_t = np.log(v_n_t)
x_tnow = x_t[-1, :]  # current value of the risk drivers

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_executive_summary-implementation-step02): Quest for Invariance

# extract the realized time series of the invariants (log-returns)
eps_t = np.diff(x_t, axis=0)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_executive_summary-implementation-step03): Estimation

# estimate sample mean and sample covariance
mu, sigma2 = meancov_sp(eps_t)
rho_1_2 = sigma2[0, 1]/np.sqrt(sigma2[0, 0]*sigma2[1, 1])

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_executive_summary-implementation-step04): Projection

# compute location and dispersion parameters
mu_x_thor = x_tnow + deltat*mu
sigma2_x_thor = deltat*sigma2

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_executive_summary-implementation-step05): Pricing

# compute parameters of bivariate normal distribution of the ex-ante P&L
v_stocks_tnow = v_n_t[-1, :2]
x_stocks_tnow = x_tnow[:2]
mu_stocks_x_thor = mu_x_thor[:2].copy()
sigma2_stocks_x_thor = sigma2_x_thor[:2, :2]
mu_pi = np.diagflat(v_stocks_tnow) @ (mu_stocks_x_thor - x_stocks_tnow)
sigma2_pi = np.diagflat(v_stocks_tnow) @ sigma2_stocks_x_thor @\
        np.diagflat(v_stocks_tnow)
rho_pi1_pi2 = sigma2_pi[0, 1]/np.sqrt(sigma2_pi[0, 0]*sigma2_pi[1, 1])

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_executive_summary-implementation-step06): Aggregation

# compute parameters of normal distribution of the portfolio ex-ante return
v_h_tnow = h @ v_stocks_tnow  # portfolio value
w_tnow = h * v_stocks_tnow / v_h_tnow  # portfolio weights
mu_r_w = w_tnow @ (mu_stocks_x_thor - x_stocks_tnow)
sigma2_r_w = w_tnow @ sigma2_stocks_x_thor @ w_tnow

# ## [Step 7](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_executive_summary-implementation-step07): Ex-ante evaluation

# compute satisfaction of the portfolio ex-ante return
sigma_r_w = np.sqrt(sigma2_r_w)
satis_r_w = -sigma_r_w

# ## [Step 8](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_executive_summary-implementation-step08): Ex-ante attribution

# +
# Step 8a: Ex-ante attribution: performance
# joint value at tnow
v_tnow = v_n_t[-1, :]
# parameters of joint P&L
mu_pi1_pi2_pisandp = np.diagflat(v_tnow) @ (mu_x_thor - x_tnow)
sigma2_pi1_pi2_pisandp = np.diagflat(v_tnow) @ sigma2_x_thor @\
        np.diagflat(v_tnow)
# parameters of joint returns
mu_r1_r2_rsandp = np.diagflat(1/v_tnow) @ mu_pi1_pi2_pisandp
sigma2_r1_r2_rsandp = np.diagflat(1/v_tnow) @ sigma2_pi1_pi2_pisandp @\
        np.diagflat(1/v_tnow)
# parameters of joint and marginal portfolio and S&P returns
b = np.array([[w_tnow[0], w_tnow[1], 0], [0, 0, 1]])
mu_r_z = b @ mu_r1_r2_rsandp
sigma2_r_z = b @ sigma2_r1_r2_rsandp @ b.T
rho_r_z = sigma2_r_z[0, 1]/np.sqrt(sigma2_r_z[0, 0]*sigma2_r_z[1, 1])
mu_z, sigma2_z = mu_r_z[-1], sigma2_r_z[-1, -1]
sigma_z = np.sqrt(sigma2_z)
mu_r = mu_r_z[0]
sigma2_r = sigma2_r_z[0, 0]
sigma_r = np.sqrt(sigma2_r)
# parameters of linear attribution model of the portfolio return
beta = rho_r_z * sigma_r / sigma_z
alpha = mu_r - beta * mu_z
# variance of the residual
sigma2_u = sigma2_r * (1 - rho_r_z ** 2)

# Step 8b: Ex-ante attribution: risk
# compute contributions of Z and U
risk_rw_z = (beta ** 2) * sigma2_z / sigma_r
risk_rw_u = sigma2_u / sigma_r
# -

# ## [Step 9](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_executive_summary-implementation-step09): Construction

# +
# Step 9a: Construction: portfolio optimization
# find minimum-variance portfolio with given budget constraint
h_star = v_h_tnow * (np.linalg.inv(sigma2_pi) @ v_stocks_tnow) /\
     (v_stocks_tnow @ np.linalg.inv(sigma2_pi) @ v_stocks_tnow)
h_star = np.floor(h_star)

# Step 9b: Construction: cross-sectional strategies
# construct a simple cross-sectional strategy
s_mom_tnow = eps_t[-1, :2]
if s_mom_tnow[0] > s_mom_tnow[1]:
    h_mom_tnow = np.array([h[0] + h[1]*v_stocks_tnow[1]/v_stocks_tnow[0], 0.0])
else:
    h_mom_tnow = np.array([0.0, h[0]*v_stocks_tnow[0]/v_stocks_tnow[1] + h[1]])
h_mom_tnow = np.round(h_mom_tnow)

# Step 9c: Construction: time series strategies
# construct more conservative strategy
h_tnow_risky = v_risky / v_h_tnow
