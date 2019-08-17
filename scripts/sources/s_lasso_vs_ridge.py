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

# # s_lasso_vs_ridge [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_lasso_vs_ridge&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-ridge-vs-lasso).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

from arpym.estimation import exp_decay_fp, fit_lfm_ridge, fit_lfm_lasso
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_lasso_vs_ridge-parameters)

k_ = 10  # number of stocks
l_ = 150  # length of penalties grid
lambda_ridge_max = 10**(-6)  # maximum value of ridge penalties
lambda_lasso_max = 2*10**(-4)  # maximum value of lasso penalties
tau_hl = 252  # half-life parameter in flexible probabilities
t_first = '2008-01-01'  # starting date
t_last = '2012-01-01'  # ending date

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_lasso_vs_ridge-implementation-step00): Upload data

# +
path = '../../../databases/global-databases/equities/db_stocks_SP500/'
spx = pd.read_csv(path + 'SPX.csv', index_col=0, parse_dates=['date'])
stocks = pd.read_csv(path + 'db_stocks_sp.csv', skiprows=[0], index_col=0)

# merging datasets
spx_stocks = pd.merge(spx, stocks, left_index=True, right_index=True)

# select data within the date range
spx_stocks = spx_stocks.loc[(spx_stocks.index >= t_first) &
                          (spx_stocks.index <= t_last)]

# remove the stocks with missing values
spx_stocks = spx_stocks.dropna(axis=1, how='any')

date = spx_stocks.index
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_lasso_vs_ridge-implementation-step01): Select stocks and SPX from database

v_stocks = np.array(spx_stocks.iloc[:, 1+np.arange(k_)])  # select stocks
v_spx = np.array(spx_stocks.iloc[:, 0])

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_lasso_vs_ridge-implementation-step02): Compute linear returns of both SPX and stocks

x = np.diff(v_spx)/v_spx[:-1]  # benchmark
z = np.diff(v_stocks, axis=0)/v_stocks[:-1, :]  # factors
t_ = len(x)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_lasso_vs_ridge-implementation-step03): Set the flexible probabilities

p = exp_decay_fp(t_, tau_hl)  # exponential decay

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_lasso_vs_ridge-implementation-step04): Perform ridge regression

lambdagrid_ridge = np.linspace(0, lambda_ridge_max, l_)  # grid of penalties
beta_r = np.zeros((k_, l_))
for l in range(l_):
    # ridge regression
    _, beta_r[:, l], _, _ = fit_lfm_ridge(x, z, p, lambdagrid_ridge[l])

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_lasso_vs_ridge-implementation-step05): Perform lasso regression

lambdagrid_lasso = np.linspace(0, lambda_lasso_max, l_)  # grid of penalties
beta_l = np.zeros((k_, l_))
for l in range(l_):
    # lasso regression
    _, beta_l[:, l], _, _ = fit_lfm_lasso(x, z, p, lambdagrid_lasso[l])

# ## Plots

# +
plt.style.use('arpm')

color = np.random.rand(k_, 3)

# reordering for visual purpose
b_r_plot = np.squeeze((beta_r.T))
b_l_plot = np.squeeze((beta_l.T))
ind_plot = np.zeros(k_)
for k in range(k_):
    ind_plot[k] = np.where(b_l_plot[:, k] > 0)[0][-1] + 1

ind_plot = np.argsort(ind_plot.flatten())
b_r_plot = b_r_plot[:, ind_plot]
b_l_plot = b_l_plot[:, ind_plot]
col = np.array(color)[ind_plot, :].squeeze()

# axis limit
l_min = np.min(beta_l)
l_max = np.max(beta_l)
r_min = np.min(beta_r)
r_max = np.max(beta_r)
mmin = np.minimum(l_min, r_min)
mmax = np.maximum(l_max, r_max)
mmin = mmin - (mmax - mmin) / 15
mmax = mmax + (mmax - mmin) / 15
if mmin >= 0:
    mmin = -(mmax - mmin) / 15
elif mmax <= 0:
        mmax = (mmax-mmin) / 15

fig, ax = plt.subplots(2, 1)
plt.sca(ax[0])
for k in range(k_):
    plt.plot(lambdagrid_ridge, b_r_plot[:, k],
             color=to_rgb(col[k].squeeze()), lw=1.5)

plt.xlabel('$\lambda_{ridge}$')
plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='x')
plt.ylabel('Ridge loadings')
plt.grid(True)
plt.xlim(np.array([lambdagrid_ridge[0], lambdagrid_ridge[-1]]))
plt.ylim([mmin, mmax])

plt.sca(ax[1])

for k in range(k_):
    plt.plot(lambdagrid_lasso, b_l_plot[:, k],
             color=to_rgb(col[k].squeeze()), lw=1.5)

plt.xlabel('$\lambda_{lasso}$')
plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='x')
plt.ylabel('Lasso loadings')
plt.grid(True)
plt.xlim([lambdagrid_lasso[0], lambdagrid_lasso[-1]])
plt.ylim([mmin, mmax])
add_logo(fig, axis=ax[0], location=1)
plt.tight_layout()
