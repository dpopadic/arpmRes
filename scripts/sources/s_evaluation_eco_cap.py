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

# # s_evaluation_eco_cap [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_evaluation_eco_cap&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=EBEconomicCapital).

# +
import numpy as np
import pandas as pd
from scipy.stats import norm

from arpym.statistics.quantile_sp import quantile_sp
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_eco_cap-parameters)

c = 0.99  # confidence level
rho_lb_a_n = 0.03
rho_ub_a_n = 0.16
lambda_a_n = 35
s_n = 25 * 1e6  # firm size
s_lb = 5 * 1e6  # minimum firm size
s_ub = 50 * 1e6  # maximum firm size
l_lb = -0.04
l_ub = 0
tau_n = 3  # maturity

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_eco_cap-implementation-step00): Load data

# +
path = '../../../databases/temporary-databases/'
df = pd.read_csv(path + 'db_aggregation_regcred.csv', index_col=None, header=0)

j_ = df['p_j'].count()  # number of scenarios of the P&L at the horizon
n_ = df['p_n'].count()  # number of counterparties

p = np.array(df['p_j'].iloc[:j_]).reshape(-1)  # scenario-probabilities
lgd_ead_n = np.array(df['loss_n'].iloc[:n_]).reshape(-1)  # losses
p_n = np.array(df['p_n'].iloc[:n_]).reshape(-1)  # probabilities of default
rho_n = np.array(df['rho_n'].iloc[:n_]).reshape(-1)  # correlation coefficients
# idiosyncratic shock senariors
inv_phi_utilde = np.array(df['inv_phi_utilde'].iloc[:j_*n_]).reshape((j_, n_))
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_eco_cap-implementation-step01): Compute the economic capital by using its definition

# +
z = np.random.normal(0, 1, j_)  # grid of values for the risk factor Z_0

inv_phi_u = np.zeros((j_, n_))
for n in range(n_):
    inv_phi_u[:, n] = z * np.sqrt(rho_n[n]) + inv_phi_utilde[:, n] * \
                np.sqrt(1 - rho_n[n])

indicator_d_n = (inv_phi_u <= norm.ppf(p_n)).astype(float)
pi_eni = - indicator_d_n @ lgd_ead_n
eco_cap = quantile_sp(c, p) + pi_eni @ p  # economic capital
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_eco_cap-implementation-step02): Compute the approximated economic capital (according to the regulatory risk framework)

aux = (norm.ppf(p_n) - np.sqrt(rho_n) * norm.ppf(1 - c)) / np.sqrt(1 - rho_n)
eco_cap_rc = lgd_ead_n @ (norm.cdf(aux) - p_n)  # approximated economic capital

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_eco_cap-implementation-step03): Compute the approximated economic capital after the correlation specification

# +
# Define the linear increasing function of the firm size


def lf(s_n, s_lb, s_ub):
    if s_n <= s_lb:
        return l_lb
    if s_n >= s_ub:
        return l_ub
    else:
        return (l_ub - l_lb) * (s_n - s_lb) / (s_ub - s_lb) + l_lb


aux1 = (1 - np.exp(-lambda_a_n * p_n)) / (1 - np.exp(-lambda_a_n))
# correlation coefficients
rho_ca = rho_lb_a_n * aux1 + rho_ub_a_n * (1 - aux1) + lf(s_n, s_lb, s_ub)
aux2 = (norm.ppf(p_n) - np.sqrt(rho_ca) * norm.ppf(1 - c))/np.sqrt(1 - rho_ca)
# approximated economic capital
eco_cap_ca = lgd_ead_n @ (norm.cdf(aux2) - p_n)
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_eco_cap-implementation-step04): Compute the approximated economic captial with the maturity adjustment

# +

def b(p_n):
    return (0.11852 - 0.05478 * np.log(p_n))**2  # smoothing function


ma_n = (1+(tau_n-2.5)*b(p_n))/(1-1.5*b(p_n))  # maturity adjustment
# approximated economic capital
eco_cap_ma = lgd_ead_n @ ((norm.cdf(aux) - p_n) * ma_n)
# -

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_eco_cap-implementation-step05): Compute the approximated derivatives of the economic capital

h_n = np.random.randint(1, 101, n_)  # portfolio holdings
eco_cap_n = -(lgd_ead_n / h_n) * (p_n - norm.cdf(aux))
