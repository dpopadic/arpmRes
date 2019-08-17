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

# # s_reg_truncated_lfm [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_reg_truncated_lfm&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-trunc-time).

# +
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from arpym.statistics import meancov_sp
from arpym.estimation import fit_lfm_ols, cov_2_corr
from arpym.tools import histogram_sp, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_reg_truncated_lfm-parameters)

spot = np.array([0, 1, 9])  # targets and factors to spot
n_long = 61  # long index
n_short = np.array([366, 244])  # short indices

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_reg_truncated_lfm-implementation-step00): Load data

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

t_ = len(data.index) - 1
n_ = len(data.columns)
k_ = len(idx_sector.columns)
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_reg_truncated_lfm-implementation-step01): Compute linear returns of X and Z

v_stock = data.values
x = (v_stock[1:, :] - v_stock[:-1, :]) / v_stock[:-1, :]
v_sector = idx_sector.values
z = (v_sector[1:, :] - v_sector[:-1, :]) / v_sector[:-1, :]

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_reg_truncated_lfm-implementation-step02): Compute OLSFP estimates and residuals

alpha, beta, s2, u = fit_lfm_ols(x, z)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_reg_truncated_lfm-implementation-step03): Compute the joint covariance and correlation

# +
# compute covariance
[mu_uz, sig2_uz] = meancov_sp(np.hstack((u, z)))
sig2_u = sig2_uz[:n_, :n_]
sig2_z = sig2_uz[n_:, n_:]

# compute correlation
c2_uz, _ = cov_2_corr(sig2_uz)
c_uz = c2_uz[:n_, n_:]
c2_u = np.tril(c2_uz[:n_, :n_], -1)
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_reg_truncated_lfm-implementation-step04): Compute standard deviations of two portfolios

# +
w_1 = np.ones(n_) / n_  # equal weight portfolio
w_2 = np.zeros(n_)  # long/shoft portfolio
w_2[n_long] = 0.69158715
w_2[n_short] = np.array([-0.67752045, -0.01406671])

_, sig2_x = meancov_sp(x)
sig2_x_trunc = beta @ sig2_z @ beta.T + np.diag(np.diag(sig2_u))

std_1 = np.sqrt(w_1.T @ sig2_x @ w_1)
std_trunc_1 = np.sqrt(w_1.T @ sig2_x_trunc @ w_1)

std_2 = np.sqrt(w_2.T @ sig2_x @ w_2)
std_trunc_2 = np.sqrt(w_2.T @ sig2_x_trunc @ w_2)
# -

# ## Plots

# +
# Compute regression plane for selected target variables and factors
beta_spot = beta[spot[0], spot[1:]]
alpha_spot = alpha[spot[0]]

m_z_spot, s2_z_spot = meancov_sp(z[:, spot[1:]])

z_grid = np.linspace(-3, 3, 6)
z_1 = m_z_spot[0] + z_grid * np.sqrt(s2_z_spot[0, 0])
z_2 = m_z_spot[1] + z_grid * np.sqrt(s2_z_spot[1, 1])
[z_1, z_2] = np.meshgrid(z_1, z_2)
x_reg = alpha_spot + beta_spot[0] * z_1 + beta_spot[1] * z_2
x_gen = alpha_spot * np.ones(x_reg.shape)

# Display regression plane, generic plane and observations of selected target
# variables and factors
plt.style.use('arpm')
fig1, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})

ax.plot_wireframe(z_2, z_1, x_reg, edgecolor='b')
ax.scatter(x[:, spot[0]], z[:, spot[1]],
           z[:, spot[2]], marker='.', color='k')
plt.legend(['regression plane'])
plt.xlabel('factor %d' % (spot[2]+1), labelpad=10)
plt.ylabel('factor %d' % (spot[1]+1), labelpad=10)
ax.set_zlabel('mkt variable %d' % (spot[0]+1), labelpad=10)

# (untruncated) correlations among residuals
corr_u = c2_u[np.nonzero(c2_u)]  # reshape the correlations
n, xout = histogram_sp(corr_u)

add_logo(fig1)
plt.tight_layout()

fig2 = plt.figure()
plt.bar(xout, n, width=xout[1]-xout[0], facecolor=[.7, .7, .7], edgecolor='k')
plt.title('Correlations among residuals')

# (untruncated) correlations between factors and residuals
corr_uz = np.reshape(c_uz, (n_*k_,), 'F')  # reshape the correlations
n, xout = histogram_sp(corr_uz)

add_logo(fig2, location=1)
plt.tight_layout()

fig3 = plt.figure()
plt.bar(xout, n, width=xout[1]-xout[0], facecolor=[.7, .7, .7], edgecolor='k')
plt.title('Correlations between factors residuals')

add_logo(fig3, location=1)
plt.tight_layout()
