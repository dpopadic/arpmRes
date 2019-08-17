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

# # s_attribution_norm [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_attribution_norm&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=EBTopDownExpN).

# +
import numpy as np
import pandas as pd

from arpym.statistics import objective_r2
from arpym.tools import forward_selection, backward_selection
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_attribution_norm-implementation-step01): Upload data

# +
path = '../../../databases/temporary-databases'
df = pd.read_csv(path + '/db_pricing_zcb.csv', header=0)
d_ = len(np.array(df['y_tnow'].dropna(axis=0, how='all')))
n_ = 2  # number of instruments
alpha_pi_pric = np.array(df['alpha_pi_pric'].dropna(axis=0,
                         how='all'))
beta_pi_pric = np.array(df['beta_pi_pric'].dropna(axis=0,
                        how='all')).reshape(d_, n_)
# expectation of the risk-drivers at horizon
mu_thor = np.array(df['mu_thor'].dropna(axis=0, how='all'))
# variance of the risk-drivers at horizon
sig2_thor = np.array(df['sig2_thor'].dropna(axis=0, how='all')).reshape(d_, d_)
mu_pl = np.array(df['mu_pl'].dropna(axis=0, how='all'))
sig2_pl = np.array(df['sig2_pl'].dropna(axis=0, how='all')).reshape(n_, n_)

path = '../../../databases/temporary-databases/'
db2 = pd.read_csv(path + 'db_aggregation_normal.csv', index_col=0)
n_ = int(np.array(db2['n_'].iloc[0]))
h = np.array(db2['h'].iloc[:n_]).reshape(-1)
mu_h = np.array(db2['mu_h'].iloc[0])
sig2_h = np.array(db2['sig2_h'].iloc[0])

path = '../../../databases/temporary-databases/'
db3 = pd.read_csv(path + 'db_cross_section.csv', index_col=0)
k_ = int(np.array(db3['k_'].iloc[0]))
alpha_pi_style = np.array(db3['alpha'].iloc[:k_+1]).reshape(-1)
beta_pi_style = np.array(db3['beta'].iloc[:k_+1]).reshape(-1,1).T
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_attribution_norm-implementation-step02): Bottom-up shift term and exposures (pricing factors)

alpha_bottomup_pric = alpha_pi_pric@h  # bottom-up shift term (pricing factors)
beta_bottomup_pric = beta_pi_pric@h  # bottom-up exposure (pricing factors)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_attribution_norm-implementation-step03): Bottom-up shift term and exposure (style factors)

alpha_bottomup_style = alpha_pi_style@h  # bottom-up shift term (style factors)
beta_bottomup_style = beta_pi_style@h  # bottom-up exposure (style factors)

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_attribution_norm-implementation-step04): Top-down shift term and exposure (two factors)

# +
# risk factors expectation (two factors)
mu_z = np.array([mu_thor[0], mu_thor[5]])

# risk factors covariance (two factors)
sig2_z = np.array([[sig2_thor[0, 0], sig2_thor[0, 5]],
                   [sig2_thor[0, 5], sig2_thor[5, 5]]])
# covariance between Pi_h and Z_1
sig_pi_z1 = np.sum((h[0] * beta_pi_pric[:, 0] +
                    h[1] * beta_pi_pric[:, 1])@sig2_thor[0, :])

# covariance between Pi_h and Z_2
sig_pi_z2 = np.sum((h[0] * beta_pi_pric[:, 0] +
                    h[1] * beta_pi_pric[:, 1])@sig2_thor[5, :])
# top-down exposures (two factors)
beta_topdown_twofactors = np.array([sig_pi_z1,
                                    sig_pi_z2])@np.linalg.inv(sig2_z)
# top-down alpha (two factors)
alpha_topdown_twofactors = mu_h - beta_topdown_twofactors@mu_z
# -

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_attribution_norm-implementation-step05): Top-down shift term and exposure (one factor)

# +
# covariance between Pi_h and Z
sig2_piz = np.array([[sig2_h, sig_pi_z1, sig_pi_z2],
                     [sig_pi_z1, sig2_z[0, 0], sig2_z[0, 1]],
                     [sig_pi_z2, sig2_z[0, 1], sig2_z[1, 1]]])
# objective function is r-squared
def objective(j):
    return objective_r2(j, sig2_piz, 1)

j_fwd = forward_selection(objective, 2)  # select factors via forward selection
j_bkd = backward_selection(objective, 2)  # select factors via backward select.

beta = sig_pi_z2/sig2_z[1, 1]  # top-down exposures (one factor)
alpha = mu_h - beta*mu_z[1]  # top-down alpha (one factor)
# -

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_attribution_norm-implementation-step06): Parameters normal distribution of (U, Z_select)

# residual variance
sig2_u = sig2_h-2*beta*sig_pi_z2 + beta*beta*sig2_z[1, 1]
mu_u_z = np.array([0, mu_z[1]])  # expectation of (U, Z_select)
sig2_u_z = np.array([[sig2_u, 0],
                     [0, sig2_z[1, 1]]])  # covariance of (U, Z_select)

# ## [Step 7](https://www.arpm.co/lab/redirect.php?permalink=s_attribution_norm-implementation-step07): Parameters normal distribution of (Z_0, Z_select)

mu_z_z = np.array([alpha, mu_z[1]])  # expectation of (Z_0, Z_select)
sig2_z_z = np.array([[sig2_u, 0],
                    [0, sig2_z[1, 1]]])  # covariance of (Z_0, Z_select)

# ## [Step 8](https://www.arpm.co/lab/redirect.php?permalink=s_attribution_norm-implementation-step08): Save data

# +
output = {
          'beta': pd.Series(beta),
          'mu_z_z': pd.Series(mu_z_z),
          'sig2_z_z': pd.Series(sig2_z_z.reshape((n_ * n_,))),
          }

df = pd.DataFrame(output)
df.to_csv('../../../databases/temporary-databases/db_attribution_normal.csv',
          index=None)
