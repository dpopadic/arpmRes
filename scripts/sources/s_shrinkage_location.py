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

# # s_shrinkage_location [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_shrinkage_location&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExStein).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arpym.tools import quad_prog, sector_select, add_logo

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_shrinkage_location-parameters)

i1 = int(0)  # select first stock
i2 = int(1)  # select second stock
gamma = 0.8  # confidence in James-Stein estimator

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_shrinkage_location-implementation-step00): Load data

# +
path = '../../../databases/global-databases/equities/'
df_stocks = pd.read_csv(path + 'db_stocks/data.csv', index_col=0,
                        parse_dates=['date'])
df_sectors = pd.read_csv(path + 'db_stocks/sectors.csv')

sectors = df_sectors.sector
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_shrinkage_location-implementation-step01): Select equities belonging to the same sector

# +
ind_stocks = sector_select(sectors, sectors[2])
names_stocks = df_sectors.loc[ind_stocks].symbol.tolist()
names_select = [names_stocks[i] for i in [i1, i2]]

v = df_stocks.loc[:, names_stocks]
v = v.dropna(axis=1, how='all')  # remove completely empty columns
v = v.dropna(axis=0, how='any')  # remove missing observations

v = np.array(v)
i_ = len(ind_stocks)
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_shrinkage_location-implementation-step02): Compute the log-returns of the selected stocks and the global mean

epsi_global = np.diff(np.log(v), axis=0)
mu = np.mean(epsi_global, axis=0)  # global sample expectation
t_global = len(epsi_global)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_shrinkage_location-implementation-step03): Randomly select the estimation sample

t_ = int(np.floor((t_global / 4)))  # length of the sample
perm = np.random.permutation(np.arange(t_global))  # random combination
epsi = epsi_global[perm[:t_], :]  # estimation sample
epsi_out = epsi_global[perm[t_:], :]

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_shrinkage_location-implementation-step04): Compute the grand mean of the sector

mu_target = np.mean(epsi)  # grand mean

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_shrinkage_location-implementation-step05): Compute the mean estimators (sample and shrinkage mean)

mu_sample = np.mean(epsi, axis=0)
mu_shrink = (1-gamma)*mu_sample + gamma*mu_target

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_shrinkage_location-implementation-step06): Compute the loss for both sample and shrinkage estimators

loss_sample = np.sum((mu_sample - mu) ** 2)
loss_shrink = np.sum((mu_shrink - mu) ** 2)

# ## [Step 7](https://www.arpm.co/lab/redirect.php?permalink=s_shrinkage_location-implementation-step07): Compute mean-variance optimal weights using the sample and shrinkage mean estimators

# +
# set the constraints and options for quad_prog
lb = np.zeros((2, 1))
ub = np.ones((2, 1))
x0 = (1/2)*np.ones((2, 1))
beq = np.array([[1]])
aeq = np.ones((1, 2))
sigma2 = np.cov(epsi[:, [i1, i2]].T, bias=True)  # sample covariance matrix

w_sample = quad_prog(sigma2, -mu_sample[[i1, i2]], aeq, beq, lb, ub, x0)
w_shrink = quad_prog(sigma2, -mu_shrink[[i1, i2]], aeq, beq, lb, ub, x0)
# -

# ## Plots

# +
plt.style.use('arpm')

c0 = [0.8, 0.8, 0.8]
c1 = [1, 0.5, 0.1]

fig1 = plt.figure()

plt.plot(epsi[:, i1], epsi[:, i2], 'd', markerfacecolor='k',
         markersize=5)
plt.xlim(np.percentile(epsi_global[:, i1], [20, 80]))
plt.ylim(np.percentile(epsi_global[:, i2], [20, 80]))
plt.plot(epsi_out[:, i1], epsi_out[:, i2], 'o', color=c0, markersize=5)
plt.plot(mu_sample[i1], mu_sample[i2], 'o', markerfacecolor=c1)
plt.plot(mu[i1], mu[i2], 'o', markerfacecolor='k')
plt.plot(mu_shrink[i1], mu_shrink[i2], 'o', markerfacecolor='g')
plt.plot([mu_sample[i1], mu[i1]], [mu_sample[i2], mu[i2]],
         color=c1)
plt.plot([mu_shrink[i1], mu[i1]], [mu_shrink[i2], mu[i2]],
         color='g')
plt.legend(['sample scenarios (%3.0f)' % t_,
            'out-of-sample scenarios (%3.0f)' % (t_global - t_),
            'sample mean', 'global mean', 'shrinkage mean'],
           bbox_to_anchor=(0., .8, 1.2, .102), frameon=True, facecolor='white')
plt.xlabel(names_select[0])
plt.ylabel(names_select[1])

add_logo(fig1)

fig2, ax = plt.subplots(2, 1)
plt.sca(ax[0])
plt.bar(1, loss_sample, 0.4, facecolor=c1, edgecolor=c1)
plt.bar(2, loss_shrink, 0.4, facecolor='g', edgecolor='g')
plt.xlim([0.5, 2.5])
plt.ylim([0, max([loss_sample, loss_shrink])])
plt.title('Loss')
plt.xticks([1, 2], ['sample', 'shrinkage'])

plt.sca(ax[1])
plt.bar(2, w_shrink[0] + w_shrink[1], 0.4, facecolor='g', edgecolor='g')
plt.bar(1, w_sample[0] + w_sample[1], 0.4, facecolor=c1, edgecolor=c1)
plt.bar(2, w_shrink[0], 0.4, facecolor='w', edgecolor='g')
plt.bar(1, w_sample[0], 0.4, facecolor='w', edgecolor=c1)
plt.xlim([.5, 2.5])
plt.ylim([0, max([np.sum(w_sample), np.sum(w_shrink)])])
plt.xticks([1, 2], ['sample', 'shrinkage'])
plt.title('Portfolio Weights')
plt.legend([names_select[1], names_select[1], names_select[0],
            names_select[0]])
add_logo(fig2, axis=ax[0])
plt.tight_layout()
