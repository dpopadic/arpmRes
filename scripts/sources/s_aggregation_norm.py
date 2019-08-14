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

# # s_aggregation_norm [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_aggregation_norm&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-normal-first-order-approx).

# +
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from arpym.tools.histogram_sp import histogram_sp
# from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_aggregation_norm-parameters)

h = np.array([100000, 80000])  # portfolio holdings

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_aggregation_norm-implementation-step01): Load data

# +
path = 'databases/temporary-databases'
df = pd.read_csv(path + '/db_pricing_zcb.csv', header=0)

j_, _ = df.shape  # number of scenarios
# number of key-rates
d_ = len(np.array(df['y_tnow'].dropna(axis=0, how='all')))
# number of instruments
n_ = len(np.array(df['v_zcb_tnow'].dropna(axis=0, how='all')))

# scenarios for the ex-ante P&L's
pl = np.array([df['pl' + str(i + 1)] for i in range(n_)]).T
# bonds' P&L's mean
mu_pl = np.array(df['mu_pl'].dropna(axis=0, how='all'))
# bonds' P&L's covariance
sig2_pl = np.array(df['sig2_pl'].dropna(axis=0, how='all')).reshape((n_, n_))

# horizon
deltat = float(df['time2hor_tnow'].dropna(axis=0, how='all'))
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_aggregation_norm-implementation-step02): Scenarios for the portfolio's P&L and its expectation and variance

pl_h = pl@h  # portfolio P&L scenarios
mu_h = mu_pl@h  # portfolio P&L expectation
sig2_h = h@sig2_pl@h  # portfolio P&L variance

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_aggregation_norm-implementation-step03): Compute the heights and bin centers of the histogram

f_pi_h, ksi = histogram_sp(pl_h, p=(1 / j_ * np.ones(j_)), k_=np.round(10 * np.log(j_)))

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_aggregation_norm-implementation-step04): Save data in database db_aggregation_normal

# +
output = {'n_': pd.Series(n_),
          'mu_h': pd.Series(mu_h),
          'sig2_h': pd.Series(sig2_h),
          'h': pd.Series(h),
         }

df = pd.DataFrame(output)
df.to_csv(path + 'db_aggregation_normal.csv')
# -

# ## Plots

# +
# plt.style.use('arpm')

fig = plt.figure()
ax = fig.add_subplot(111)
lgray = [.8, .8, .8]  # light gray
dgray = [.7, .7, .7]  # dark gray

plt.bar(ksi, f_pi_h, width=ksi[1] - ksi[0],
        facecolor=lgray, edgecolor=dgray)
plt.title(r"Distribution of the portfolio's P&L " +
          "at the horizon ($\Delta t=${horizon:.0f} days)"
          .format(horizon=deltat * 252))

x_hor = np.linspace(mu_h - 7 * np.sqrt(sig2_h),
                    mu_h + 7 * np.sqrt(sig2_h), 500)
taylor_first = norm.pdf(x_hor, loc=mu_h, scale=np.sqrt(sig2_h))

plt.plot(x_hor, taylor_first.flatten(), 'r', lw=1.5)
ax.set_xlim([x_hor[0], x_hor[-1]])
plt.legend(['Normal approx'])

# add_logo(fig)
plt.tight_layout()
