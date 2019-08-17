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

# # s_shrink_spectrum_filt [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_shrink_spectrum_filt&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=SpectrumShrinkage).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arpym.estimation import spectrum_shrink
from arpym.tools import histogram_sp, pca_cov, add_logo

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_shrink_spectrum_filt-parameters)

t_first = '2007-01-01'  # starting date
t_last = '2012-01-01'  # ending date

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_shrink_spectrum_filt-implementation-step00): Load data

# +
# upload stocks values
path = '../../../databases/global-databases/equities/db_stocks_SP500/'
df_stocks = pd.read_csv(path + 'db_stocks_sp.csv', index_col=0, header=[0, 1])

# select data within the date range
df_stocks = df_stocks.loc[(df_stocks.index >= t_first) &
                          (df_stocks.index <= t_last)]

# remove the stocks with missing values
df_stocks = df_stocks.dropna(axis=1, how='any')

v = np.array(df_stocks)
i_ = v.shape[1]
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_shrink_spectrum_filt-implementation-step01): Compute the log-returns

epsi = np.diff(np.log(v), axis=0)  # log-returns
t_ = epsi.shape[0]

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_shrink_spectrum_filt-implementation-step02): Standardize time series of invariants

# standardized invariants
epsi_tilde = (epsi - np.mean(epsi, axis=0)) / np.std(epsi, axis=0)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_shrink_spectrum_filt-implementation-step03): Compute the sample covariance matrix and its eigenvalues

sigma2_hat = np.cov(epsi_tilde.T)  # sample covariance
_, lambda2_hat = pca_cov(sigma2_hat)  # sample spectrum

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_shrink_spectrum_filt-implementation-step04): Perform spectrum shrinkage

sigma2_bar, lambda2_bar, k_, err, y_mp, x_mp, dist = \
                                                spectrum_shrink(sigma2_hat, t_)

# ## Plots

# +
plt.style.use('arpm')

c0_bl = [0.27, 0.4, 0.9]
c1_or = [1, 0.5, 0.1]

# spectrum plot
fig1 = plt.figure()
plt.bar(np.arange(i_), np.log(lambda2_hat), facecolor=c0_bl,
        label='sample spectrum')
plt.plot(np.arange(k_), np.log(lambda2_bar[:k_]), color=c1_or, lw=2)
plt.plot(np.arange(k_, i_), np.log(lambda2_bar[k_:i_]), color=c1_or, lw=2,
         label='filtered spectrum')
plt.legend()
plt.title('Spectrum')
plt.ylabel('log-eigenvalues')
plt.xlabel('stocks')
add_logo(fig1, location=5)

# spectrum distribution
fig2 = plt.figure()
density, xbin = histogram_sp(lambda2_hat, k_=10*i_)
pp1 = plt.bar(xbin, density, width=xbin[1]-xbin[0], facecolor=c0_bl,
              edgecolor='none', label='sample eigenvalues below threshold')
pp2 = plt.plot(x_mp, y_mp*(1 - k_ / i_), color='g', lw=2,
               label='Marchenko-Pastur fit')
x_filtered = lambda2_bar[:k_ + 2]
density_filtered = np.r_['-1', np.ones((1, k_+1)), np.array([[i_ - k_]])]
pp3 = plt.plot(np.r_[x_filtered.reshape(1, -1), x_filtered.reshape(1, -1)],
               np.r_[np.zeros((1, k_ + 2)), density_filtered], color=c1_or,
               lw=2, label='filtered spectrum')
plt.xlim([0, 3*np.max(x_mp)])
plt.ylim([0, max([np.max(y_mp*(1 - k_ / i_)), np.max(density)])])
plt.legend(handles=[pp1, pp2[0], pp3[0]])
plt.title('Spectrum distribution')
plt.xlabel('eigenvalues')

add_logo(fig2, location=5)
