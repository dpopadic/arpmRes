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

# # s_pca_empirical [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_pca_empirical&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-swap-emp-i).

# +
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from arpym.tools import pca_cov, plot_ellipsoid, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_pca_empirical-parameters)

k_ = 10
idx = [0, 4, 8]  # target indices
r = 3  # standard deviation size

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_pca_empirical-implementation-step00): Import data

data = pd.read_csv('../../../databases/temporary-databases/db_pca_yield_tools.csv')
n_ = int(data['n_'][0])
tau = data['tau'].values[:n_]
m_hat_HFP_x = data['m_hat_HFP_x'].values[:n_]
s2_hat_HFP_x = pd.read_csv('../../../databases/temporary-databases/db_pca_yield_tools.csv',
                   usecols=['s2_hat_HFP_x']).values.reshape(n_, n_)
s_vec = data['s_vec'].values[:n_]
yields = pd.read_csv('../../../databases/temporary-databases/db_pca_yield_data.csv')
l_ = int(yields['l_'][0])
y = pd.read_csv('../../../databases/temporary-databases/db_pca_yield_data.csv',
                   usecols=['y']).values.reshape(l_, n_)

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_pca_empirical-implementation-step01): Compute eigenvectors, eigenvalues and r2

e_hat, lambda2_hat = pca_cov(s2_hat_HFP_x)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_pca_empirical-implementation-step02): Compute factor shifting

# +
y_pred = []
y_temp = np.zeros((n_, 3))

for k in range(k_):
    shift = r * np.sqrt(lambda2_hat[k]) * e_hat[:, k]
    y_temp = np.zeros((n_, 3))
    y_temp[:, 0] = y[0, :]
    y_temp[:, 1] = y[0, :] + shift
    y_temp[:, 2] = y[0, :] - shift
    y_pred.append(y_temp)
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_pca_empirical-implementation-step03): Save the data

# +
output = {
          'n_': pd.Series(len(tau)),
          'k_': pd.Series(k_),
          'e_hat': pd.Series(e_hat.reshape(-1)),
          'lambda2_hat': pd.Series(lambda2_hat),
          }

df = pd.DataFrame(output)
df.to_csv('../../../databases/temporary-databases/db_pca_empirical.csv')
# -

# ## Plots

# +

plt.style.use('arpm')

fig2, ax = plt.subplots(3, 1)
ax[0].plot(tau, y_pred[0], linewidth=1.5, color='k')
ax[0].set_title('1st factor: shift')

ax[1].plot(tau, y_pred[1], linewidth=1.5, color='k')
ax[1].set_title('2nd factor: steepening')
ax[1].set_ylabel('yield curve')

ax[2].plot(tau, y_pred[2], linewidth=1.5, color='k')
ax[2].set_title('3rd factor: bending')
ax[2].set_xlabel('time to maturity (yrs)')

add_logo(fig2, axis=ax[2], location=4)
plt.tight_layout()

alpha = np.linspace(0, 2 * np.pi, 50)
beta = np.linspace(np.pi/2, np.pi, 50)

fig3, ax = plot_ellipsoid(m_hat_HFP_x[idx],
                          s2_hat_HFP_x[np.ix_(idx, idx)], r=3,
                          alpha=alpha, beta=beta,
                          plot_axes=False, point_color=(.3, .3, .3))
ax.view_init(29, -121)
ax.set_xlabel('changes in 2yr yields', labelpad=15)
ax.set_ylabel('changes in 6yr yields', labelpad=15)
ax.set_zlabel('changes in 10yr yields', labelpad=15)
ax.invert_yaxis()

add_logo(fig3, location=4)
plt.tight_layout()
