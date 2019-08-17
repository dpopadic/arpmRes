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

# # s_risk_attrib_torsion [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_risk_attrib_torsion&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ex-minimum-torsion-vs-traditional).

# +
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from arpym.estimation import spectrum_shrink
from arpym.portfolio import minimum_torsion
from arpym.portfolio import effective_num_bets
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_risk_attrib_torsion-parameters)

delta_t = 21  # time interval between scenarios
k_ = 250  # number of factors
t_ = 252  # size of trailing window
t_star = 601  # number of observations

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_risk_attrib_torsion-implementation-step00): Load data

# +
path = '../../../databases/global-databases/equities/db_stocks_SP500/'
data = pd.read_csv(path + 'db_stocks_sp.csv', index_col=0)

prices = np.array(data.iloc[-t_star:, :k_].apply(pd.to_numeric))
dates = data.index[-t_star:]
t_star = t_star - 1  # number of max daily returns
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_risk_attrib_torsion-implementation-step01): Computing the minimum-torsion distribution and the relative marginal distribution

# +
beta = np.ones((k_, )) / (k_)  # equally weighted exposures to factors

print('Computing the minimum-torsion distribution and the relative marginal distribution')
j_ = int((t_star-t_) / delta_t) + 1  # number of scenarios
enb_mt = np.zeros(j_)
datestr = []
datenum = np.zeros(j_)
linret = {}
m = np.zeros((j_, k_))
p_mt = np.zeros((j_, k_))
for j in range(j_):
    t = t_ + j * delta_t

    # date empirical tests
    datenum[j] = int(time.mktime(time.strptime(dates[t], '%Y-%m-%d')))
    datestr.append(dates[t])

    # linear returns/factors scenarios
    linret[j] = prices[j*delta_t+1:t_+j*delta_t+1, :] /\
        prices[j*delta_t:t_+j*delta_t, :] - 1

    # sample covariance matrix
    sigma2 = np.cov(linret[j], rowvar=False)

    # spectrum shrinkage of the correlation matrix
    sigma2 = spectrum_shrink(sigma2, t_)[0]

    # minimum-torsion matrix and minimum-torsion exposures
    t_mt = minimum_torsion(sigma2)
    beta_mt = beta.dot(np.linalg.solve(t_mt, np.eye(k_)))

    # minimum-torsion diversification distribution and minimum-torsion
    # effective number of bets
    enb_mt[j], p_mt[[j], :] = effective_num_bets(beta, sigma2, t_mt)

    # marginal risk contribution (traditional approach)
    m[[j], :] = beta.T*(sigma2@(beta.T))/(beta@sigma2@(beta.T))
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_risk_attrib_torsion-implementation-step02): Sort the minimum-torsion diversification distribution and the relative marginal contributions

p = np.r_['-1', p_mt[..., np.newaxis], m[..., np.newaxis]]
weight = np.zeros((2, k_, j_))
prob = np.zeros((2, k_, j_))
for i in range(2):
    for j in range(j_):
        prob[i, :, j], index = np.sort(p[j, :, i])[::-1],\
            np.argsort(p[j, :, i])[::-1]
        weight[i, :, j] = beta[index]

# ## Plots

# +
plt.style.use('arpm')

for i in range(2):
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
    ax.view_init(18, 34)
    plt.xlim([datenum[0], datenum[-1]])
    plt.ylim([0, k_ + 1])
    plt.xlabel('time', labelpad=20)
    plt.ylabel('stocks', labelpad=10)
    ax.set_zlabel('weights/probabilities', labelpad=10)
    hsurf1 = ax.plot_surface(np.tile(datenum[np.newaxis, ...], (k_, 1)),
                             np.tile(np.arange(k_)[..., np.newaxis], (1, j_)),
                             prob[i, :, :], cmap='gray', shade=False)
    hsurf2 = ax.plot_surface(np.tile(datenum[np.newaxis, ...], (k_, 1)),
                             np.tile(np.arange(k_)[..., np.newaxis], (1, j_)),
                             weight[i, :, :], color='gray', shade=False)
    indd = np.linspace(0, len(datenum) - 1, 6, dtype=int)
    dateticks = []
    for d in indd:
        dateticks.append(time.strftime('%b-%y', time.localtime(datenum[d])))
    plt.xticks(datenum[indd], dateticks)

    if i == 0:
        plt.title('Minimum-torsion diversification distribution')
        ax.set_zlim([0, 10 ** -2 + 0.001])
        ax.plot(datenum, np.zeros(len(datenum)),
                enb_mt.flatten() / (k_) * 10 ** -2, lw=2, color='r')
        ax.plot(datenum, np.zeros(len(datenum)),
                np.ones(enb_mt.shape[0]) * 10 ** -2, lw=1, color='r')
        ax.text(datenum[0], 10, 10**-2, '1', color='r')
    else:
        plt.title('Relative marginal distribution')
        ax.set_zlim([-0.001, 10 ** -2 + 0.001])

    add_logo(fig)
    plt.tight_layout()
