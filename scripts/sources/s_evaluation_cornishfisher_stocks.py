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

# # s_evaluation_cornishfisher_stocks [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_evaluation_cornishfisher_stocks&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-corn-fish-vs-mc).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arpym.statistics import moments_logn, cornish_fisher, quantile_sp
from arpym.tools import histogram_sp, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_cornishfisher_stocks-parameters)

h = np.array([5000, 5000])  # portfolio holdings
c = 1 - np.arange(.001, 1, .001)  # confidence levels grid

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_cornishfisher_stocks-implementation-step00): Upload data

# +
# upload database generated from s_pricing_stocks_norm
path = '../../../databases/temporary-databases/'
db = pd.read_csv(path + 'db_stocks_normal.csv')

n_ = int(np.array(db['n_'].iloc[0]))
j_ = int(np.array(db['j_'].iloc[0]))
# parameters of the shifted lognormal distribution
v_tnow = np.array(db['v_tnow'].iloc[:n_]).reshape(-1)
mu_pl = np.array(db['mu_tau'].iloc[:n_]).reshape(-1)
sig2_pl = np.array(db['sigma2_tau'].iloc[:n_*n_]).reshape((n_, n_))
# Monte Carlo scenarios for the ex-ante P&L's
pl = np.array(db['pl']).reshape((j_, n_))
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_cornishfisher_stocks-implementation-step01): Ex ante P&L mean, standard deviation and skewness

mu_pl_h, sd_pl_h, sk_pl_h = moments_logn(h, mu_pl, sig2_pl, v_tnow)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_cornishfisher_stocks-implementation-step02): Cornish-Fisher approximation

q_cf = cornish_fisher(mu_pl_h, sd_pl_h, sk_pl_h, 1 - c)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_cornishfisher_stocks-implementation-step03): Scenario-probability quantile

# +
# Monte carlo scenarios for the portfolio's P&L
pl_h = h@pl.T

# scenario-probability quantile
q_sp = quantile_sp(1 - c, pl_h)
# -

# ## Plots

# +
plt.style.use('arpm')
lgray = [.8, .8, .8]  # light gray
dgray = [.7, .7, .7]  # dark gray
fig = plt.figure()

# histogram of the portfolio's ex-ante P&L
j_ = pl_h.shape[0]
n_bins = np.round(10 * np.log(j_))  # number of histogram bins
y_hist, x_hist = histogram_sp(pl_h, p=1 / j_ * np.ones(j_), k_=n_bins)

# Cornish-Fisher quantile approximation and scenario-probability quantile
l1 = plt.plot(q_sp, 1 - c, 'b')
l2 = plt.plot(q_cf, 1 - c, 'r', linestyle='--', lw=1)
l3 = plt.bar(x_hist, y_hist / max(y_hist), width=x_hist[1] - x_hist[0],
             facecolor=lgray, edgecolor=dgray)
plt.xlim([np.min(q_cf), np.max(q_cf)])
plt.ylim([0, 1])  # set 'ylim' to [0, 0.1] to focus on the left tail only
leg = plt.legend(['MC quantile', 'Cornish Fisher approx', 'MC distribution'])
plt.title('Monte Carlo quantile and Cornish-Fisher approximation')
add_logo(fig)
