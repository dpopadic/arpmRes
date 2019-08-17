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

# # s_risk_attribution_norm [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_risk_attribution_norm&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=EBNumerRiskAttrNorm).

# +
import numpy as np
import pandas as pd
from math import factorial

from scipy.special import erfinv
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_risk_attribution_norm-parameters)

c = 0.95  # confidence level

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_risk_attribution_norm-implementation-step01): Load data

# +
path = '../../../databases/temporary-databases/'
df = pd.read_csv(path + 'db_attribution_normal.csv')
beta = np.array(df['beta'].dropna(axis=0, how='all'))  # exposures
# expectation of the risk factors (Z_0,Z_1)
mu_z = np.array(df['mu_z_z'].dropna(axis=0, how='all'))
n_ = len(mu_z)
# covariance of the risk factors (Z_0,Z_1)
sig2_z = np.array(df['sig2_z_z'].dropna(axis=0, how='all')).reshape(n_, n_)

path = '../../../databases/temporary-databases/'
db = pd.read_csv(path + 'db_evaluation_satis_normal.csv')
sd_pi = int(np.array(db['-sig_pi_h'].iloc[0]))
cvar_pi_h = int(np.array(db['cvar_pi_h'].iloc[0]))
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_risk_attribution_norm-implementation-step02): First-in/isolated marginal contributions

beta = np.array([1, float(beta)])
satis_bkzk = -np.abs(beta)*np.sqrt(np.diag(sig2_z)).T
gamma_isol = sd_pi / np.sum(satis_bkzk)  # "first in" normalization constant
satis_k_isol = gamma_isol*satis_bkzk  # "first in" proportional contributions

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_risk_attribution_norm-implementation-step03): Last-in marginal contributions

first = sd_pi + beta[1]*np.sqrt(sig2_z[1, 1])
second = sd_pi + beta[0]*np.sqrt(sig2_z[0, 0])
gamma_last = sd_pi / (first + second)  # "last in" normalization constant
# "last in" proportional contributions
sd_last_0 = first*gamma_last
sd_last_1 = second*gamma_last

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_risk_attribution_norm-implementation-step04): sequential risk contributions

# +
index = [1, 0]
beta_perm = beta[index]
sig2_z_perm = sig2_z[:, index][index]

# sequential risk contributions
sd_seq = np.zeros(n_)
sd_seq[0] = -np.sqrt(beta_perm[0] * sig2_z_perm[0, 0] * beta_perm[0])
for k in range(1, n_):
    sd_seq[k] = -np.sqrt(beta_perm[:k+1] @ sig2_z_perm[:k+1, :k+1]
                         @ beta_perm[:k+1].T) +\
                np.sqrt(beta_perm[:k] @
                        sig2_z_perm[:k, :k] @ beta_perm[:k].T)
# -

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_risk_attribution_norm-implementation-step05): Shapley risk contributions

# +
def ncr(n, r):
        return factorial(n-r)*factorial(r-1) / factorial(n)

j0 = [[0], [0, 1]]
j1 = [[1], [0, 1]]

# Shapley risk contributions
satis_shapley_0 = -ncr(n_, len(j0[0])) * np.sqrt(beta[0]**2*sig2_z[0, 0]) + \
                 ncr(n_, len(j0[1])) * (-np.sqrt(beta@sig2_z@beta.T) +
                                        np.sqrt(beta[1]**2*sig2_z[1, 1]))
satis_shapley_1 = -ncr(n_, len(j1[0])) * np.sqrt(beta[1]**2*sig2_z[1, 1]) + \
                 ncr(n_, len(j1[1])) * (-np.sqrt(beta@sig2_z@beta.T) +
                                        np.sqrt(beta[0]**2*sig2_z[0, 0]))
# -

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_risk_attribution_norm-implementation-step06): Euler marginal contributions: standard deviation

ss = sig2_z@beta.T
# st. dev. Euler contributions
sd_euler_0 = -beta[0]*ss[0]/sd_pi
sd_euler_1 = -beta[1]*ss[1]/sd_pi

# ## [Step 7](https://www.arpm.co/lab/redirect.php?permalink=s_risk_attribution_norm-implementation-step07): Euler marginal contributions: variance

# variance Euler contributions
var_euler_0 = -beta[0]*ss[0]
var_euler_1 = -beta[1]*ss[1]

# ## [Step 8](https://www.arpm.co/lab/redirect.php?permalink=s_risk_attribution_norm-implementation-step08): Euler marginal contributions: cVaR

# +
integral = -1 / (np.sqrt(2*np.pi))*np.exp(-(erfinv(1 - 2*c)) ** 2)

# marginal contributions (cVaR)
es_euler_0 = beta[0]*mu_z[0] + beta[0] *\
             (sig2_z@beta.T)[0]/np.sqrt(beta@sig2_z@(beta.T))/(1 - c)*integral
es_euler_1 = beta[1]*mu_z[1] + beta[1] *\
             (sig2_z@beta.T)[1]/np.sqrt(beta@sig2_z@(beta.T))/(1 - c)*integral
