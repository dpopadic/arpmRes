#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # s_lfm_executive_summary [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_lfm_executive_summary&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_executive_summary_lfm).

# +
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from cvxopt import solvers, matrix

from arpym.estimation import fit_state_space
from arpym.statistics import meancov_sp
from arpym.views import min_rel_entropy_sp
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_lfm_executive_summary-parameters)

h = np.array([2*10**6, 8*10**5])  # vector of holdings
lambda_lasso = 1/1e6  # Lasso penalty
t_first = pd.to_datetime('16-03-2012')  # first considered date
t_now = pd.to_datetime('30-03-2012')  # last considered date
z_pstat = -0.05  # point statement
sig_view = 1/np.sqrt(252)*0.12  # partial view statement

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_lfm_executive_summary-implementation-step00): Load data

# +
path = '../../../databases/global-databases/equities/'
data = pd.read_csv(path+'db_stocks_SP500/SPX.csv',
                   parse_dates=True, index_col=0)
v_sandp = data[(data.index >= t_first) &
               (data.index <= t_now)].values.reshape(-1)
data1 = pd.read_csv(path+'db_stocks_SP500/db_stocks_sp.csv',
                   parse_dates=True, index_col=0, header=1,
                   usecols=['name', 'CVC', 'AON'])
v_stocks = data1[(data1.index >= t_first) &
                 (data1.index <= t_now)].values[:, [1, 0]]

del data, data1


# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_lfm_executive_summary-implementation-step01): Compute portfolio returns, S&P index returns and loadings of regression LFM

# returns of the 2 stocks
r_n_t = v_stocks[1:, :]/v_stocks[:-1, :] - 1
# curent portfolio value
v_h_tnow = v_stocks[-1, :].dot(h)
# portfolio weights
w_tnow = v_stocks[-1, :2]*h/v_h_tnow
# portfolio returns
x_t = np.sum(w_tnow*r_n_t, axis=1)
# S&P 500 returns
z_t = v_sandp[1:]/v_sandp[:-1] - 1
# LFM parameters
m_xz, s2_xz = meancov_sp(np.array([x_t, z_t]).T)
beta_reg = s2_xz[0, 1]/s2_xz[1, 1]
alpha_reg = m_xz[0]-beta_reg*m_xz[1]
x_pred = alpha_reg + beta_reg*z_pstat

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_lfm_executive_summary-implementation-step02): Fit logistic model and yield prediction for last observation

x_t_plus_1_binary = (x_t[1:] > 0).astype(int)
p = np.count_nonzero(x_t_plus_1_binary)/len(x_t_plus_1_binary)
logistic = LogisticRegression(penalty='l2', C=np.inf, class_weight='balanced',
                             solver='lbfgs', random_state=1, fit_intercept=1)
poly = PolynomialFeatures(degree=3, include_bias=False)
z_cubic = poly.fit_transform(z_t[:-1].reshape(-1, 1))
logistic = logistic.fit(z_cubic, x_t_plus_1_binary)
beta0_logit, beta_logit = logistic.intercept_, logistic.coef_[0]
# conditional probability predicted from last observation
p_beta_logit = logistic.predict_proba(z_cubic[[-1], :])[0, 1]

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_lfm_executive_summary-implementation-step03): Perform generalized probabilistic inference

annualized_vol = np.sqrt(s2_xz[1, 1])*np.sqrt(252)
p_base = np.ones(z_t.shape[0]) / z_t.shape[0]
mu_base = z_t @ p_base
z_ineq = -np.atleast_2d(z_t**2)
mu_view_ineq = -np.atleast_1d(sig_view ** 2 + mu_base ** 2)
z_eq = np.atleast_2d(z_t)
mu_view_eq = np.atleast_1d(mu_base)
p_upd = min_rel_entropy_sp(p_base, z_ineq, mu_view_ineq, z_eq, mu_view_eq,
                           normalize=False)

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_lfm_executive_summary-implementation-step04): Fit linear state-space model

h_t = fit_state_space(z_t, k_=1, p=p_upd)[0]

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_lfm_executive_summary-implementation-step05): Fit logistic model with Lasso penalty

C = 1/lambda_lasso
logistic_lasso = LogisticRegression(penalty='l1', C=C, class_weight='balanced',
                                   solver='liblinear', random_state=1,
                                   fit_intercept=1, max_iter=15000)
logistic_lasso = logistic_lasso.fit(z_cubic, x_t_plus_1_binary)
beta0_logit_lambda = logistic_lasso.intercept_
beta_logit_lambda = logistic_lasso.coef_[0]
# conditional probability predicted from last observation
p_beta_logit_lambda = logistic_lasso.predict_proba(z_cubic[[-1], :])[0, 1]
