#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # s_yield_change_correlation [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_yield_change_correlation&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_yield_change_correlation).

# +
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from arpym.statistics import meancov_sp
from arpym.estimation import cov_2_corr, min_corr_toeplitz
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_yield_change_correlation-parameters)

tau = np.arange(2, 10.25, 0.25) #  times to maturity

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_yield_change_correlation-implementation-step00): Import data from database

path = '../../../databases/global-databases/fixed-income/db_yields/'
df = pd.read_csv(path + 'data.csv', index_col=0)
y = np.array(df.loc[:, tau.astype('str')])
y = y[1800:, ]  # remove missing data
fx_df = pd.read_csv(path + 'data.csv', usecols=['dates'],
                    parse_dates=['dates'])
fx_df = fx_df[1801:]

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_yield_change_correlation-implementation-step01): Compute invariants

x = np.diff(y, n=1, axis=0)
t_, n_ = x.shape

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_yield_change_correlation-implementation-step02): Compute HFP mean, covariance, correlation and vector of standard deviations

m_hat_HFP_x, s2_hat_HFP_x = meancov_sp(x)
c2_HFP_x, s_vec = cov_2_corr(s2_hat_HFP_x)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_yield_change_correlation-implementation-step03): Fit and compute the Toeplitz cross-diagonal form

c2_star, gamma_star = min_corr_toeplitz(c2_HFP_x, tau)

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_yield_change_correlation-implementation-step04): Save the data

# +
output = {
          'tau': pd.Series(tau),
          'n_': pd.Series(x.shape[1]),
          'gamma_star': pd.Series(gamma_star),
          'm_hat_HFP_x': pd.Series(m_hat_HFP_x),
          's2_hat_HFP_x': pd.Series((s2_hat_HFP_x.reshape(-1))),
          's_vec': pd.Series(s_vec),
          }

df = pd.DataFrame(output)
df.to_csv('../../../databases/temporary-databases/db_pca_yield_tools.csv')

output = {
          'y': pd.Series((y.reshape(-1))),
          'l_': pd.Series(t_+1),
          }

df = pd.DataFrame(output)
df.to_csv('../../../databases/temporary-databases/db_pca_yield_data.csv')
# -

# ## Plots

# +
plt.style.use('arpm')
tau_vec = np.arange(5, 10.25, 0.5)
mydpi = 72.0
f = plt.figure(figsize=(1280.0/mydpi, 720.0/mydpi), dpi=mydpi)

gs = gridspec.GridSpec(2, 2)
gs.update(wspace=0.5, hspace=0.5)
ax1 = plt.subplot(gs[0, :])
colormap = plt.cm.gist_gray
plt.gca().set_prop_cycle(color=[colormap(i) for i in np.linspace(0, 0.5, 3)])
plt.plot(fx_df, x[:, 0], linewidth=0.5, label='5 yrs')
plt.plot(fx_df, x[:, 4], linewidth=0.5, label='7 yrs')
plt.plot(fx_df, x[:, 5], linewidth=0.5, label='10 yrs')
plt.legend(loc='lower right')
ax1.set_xlabel('Date')
ax1.set_ylabel('Observed values')
ax1.set_title('Changes in yields')

ax2 = plt.subplot(gs[1, 0])
im1 = plt.imshow(c2_HFP_x[:len(tau_vec), :len(tau_vec)], cmap='RdGy')
plt.colorbar(im1, fraction=0.046, pad=0.04)
plt.yticks(np.arange(len(tau_vec)), tau_vec)
plt.xticks(np.arange(len(tau_vec)), tau_vec)
ax2.set_title('Empirical correlation')

ax3 = plt.subplot(gs[1, 1])
im2 = plt.imshow(c2_star[:len(tau_vec), :len(tau_vec)], cmap='RdGy')
plt.colorbar(im2, fraction=0.046, pad=0.04)
plt.yticks(np.arange(len(tau_vec)), tau_vec)
plt.xticks(np.arange(len(tau_vec)), tau_vec)
ax3.set_title('Fitted correlation')

add_logo(f, ax1, location=2)
# -


