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

# # s_cross_section_truncated_lfm [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_cross_section_truncated_lfm&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-trunc-cross-section).

# +
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from arpym.estimation import cov_2_corr
from arpym.statistics import meancov_sp
from arpym.tools import histogram_sp, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_cross_section_truncated_lfm-parameters)

long_idx = 200  # long stock index
short_idx = 183  # short stock index

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_cross_section_truncated_lfm-implementation-step00): Load data

# +
path = '../../../databases/global-databases/equities/db_stocks_SP500/'
data = pd.read_csv(path + 'db_stocks_sp.csv', index_col=0, header=[0, 1],
                   parse_dates=True)
idx_sector = pd.read_csv(path + 'db_sector_idx.csv', index_col=0,
                         parse_dates=True)
idx_sector = idx_sector.drop("RealEstate", axis=1)  # delete RealEstate

dates = np.intersect1d(data.index, idx_sector.index)
data = data.loc[dates]
idx_sector = idx_sector.loc[dates]

beta = [(data.columns.get_level_values(0)[i] == idx_sector.columns).astype(int)
        for i in range(len(data.columns.get_level_values(1)))]
beta = np.array(beta)
t_ = len(dates)-1
n_, k_ = beta.shape
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_cross_section_truncated_lfm-implementation-step01): Compute linear returns of X and Z

v_stock = data.values
x = (v_stock[1:, :] - v_stock[:-1, :]) / v_stock[:-1, :]
v_sector = idx_sector.values
z = (v_sector[1:, :] - v_sector[:-1, :]) / v_sector[:-1, :]

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_cross_section_truncated_lfm-implementation-step02): Compute extraction matrix, projector matrix and shift parameter

mu_x, sig2_x = meancov_sp(x)
beta_ = beta.T / np.diag(sig2_x)
gamma = np.linalg.solve(beta_ @ beta, beta_)
proj = beta @ gamma
alpha_cs = mu_x - proj @ mu_x

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_cross_section_truncated_lfm-implementation-step03): Compute cross-sectional factors and residuals

z_cs = x @ gamma.T
u_cs = x - alpha_cs - z_cs @ beta.T

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_cross_section_truncated_lfm-implementation-step04): Estimate correlations between exogenous and cross-sectional factors

_, sig2_zz = meancov_sp(np.hstack((z, z_cs)))
c2_zz, _ = cov_2_corr(sig2_zz)  # joint correlation
c_zz = np.diag(c2_zz[:k_, k_:])

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_cross_section_truncated_lfm-implementation-step05): Compute the joint covariance and correlation

# +
mu_uz, sig2_uz = meancov_sp(np.hstack((u_cs, z_cs)))
sig2_u = sig2_uz[:n_, :n_]
sig2_z = sig2_uz[n_:, n_:]

c2_uz, _ = cov_2_corr(sig2_uz)
c_uz = c2_uz[:n_, n_:]
c2_u = np.tril(c2_uz[:n_, :n_], -1)
# -

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_cross_section_truncated_lfm-implementation-step06): Compute the risk premia

alpha_hat = mu_x
lambda_hat = gamma @ mu_x

# ## [Step 7](https://www.arpm.co/lab/redirect.php?permalink=s_cross_section_truncated_lfm-implementation-step07): Compute standard deviations of two portfolios

# +
w_1 = np.ones(n_) / n_  # equal weight portfolio
w_2 = np.zeros(n_)  # long short portfolio
w_2[long_idx] = 2
w_2[short_idx] = -1

sig2_x_trunc = beta @ sig2_z @ beta.T + np.diag(np.diag(sig2_u))

std_1 = np.sqrt(w_1.T @ sig2_x @ w_1)
std_trunc_1 = np.sqrt(w_1.T @ sig2_x_trunc @ w_1)

std_2 = np.sqrt(w_2.T @ sig2_x @ w_2)
std_trunc_2 = np.sqrt(w_2.T @ sig2_x_trunc @ w_2)
# -

# ## Plots

# +
plt.style.use('arpm')

# (untruncated) correlations among residuals
fig1 = plt.figure()
f, xi = histogram_sp(c2_u[np.nonzero(c2_u)])
plt.bar(xi, f, width=xi[1]-xi[0], facecolor=[.7, .7, .7], edgecolor='k')
plt.title('Correlations among residuals')
add_logo(fig1)

# (untruncated) correlations between factors and residuals
fig2 = plt.figure()
f, xi = histogram_sp(c_uz.reshape((n_*k_,)))
plt.bar(xi, f, width=xi[1]-xi[0], facecolor=[.7, .7, .7], edgecolor='k')
plt.title('Correlations between factors residuals')
add_logo(fig2, location=1)

plt.tight_layout()
