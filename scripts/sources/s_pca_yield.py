#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # s_pca_yield [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_pca_yield&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-swap-cont).

# +
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from arpym.tools import add_logo
# -

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_pca_yield-implementation-step00): Import data

data = pd.read_csv('../../../databases/temporary-databases/db_pca_yield_tools.csv')
n_ = int(data['n_'][0])
tau = data['tau'].values[:n_]
s2_hat_HFP_x = pd.read_csv('../../../databases/temporary-databases/db_pca_yield_tools.csv',
                   usecols=['s2_hat_HFP_x']).values.reshape(n_, n_)
s_vec = data['s_vec'].values[:n_]
gamma_star = data['gamma_star'][0]
data_empirical = pd.read_csv('../../../databases/temporary-databases/db_pca_empirical.csv')
k_ = int(data_empirical['k_'][0])
lambda2_hat = data_empirical['lambda2_hat'].values[:n_]
e_hat = pd.read_csv('../../../databases/temporary-databases/db_pca_empirical.csv',
                   usecols=['e_hat']).values.reshape(n_, n_)

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_pca_yield-implementation-step01): Compute theoretical eigenvalues

s2 = np.mean(s_vec)**2  # average volatility
omega = np.pi/len(tau)*np.linspace(1, k_, k_)  # frequences
lambda2_omega = 2 * s2 * gamma_star / (gamma_star**2 + omega**2)  # eigenvalues

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_pca_yield-implementation-step02): Compute theoretical and empirical r squared

r2_omega = 2 / np.pi * np.arctan(omega / gamma_star)  # theoretical
r2_hat = np.cumsum(lambda2_hat) / np.sum(lambda2_hat)  # empirical

# ## Plots

# +

plt.style.use('arpm')

darkred = [.9, 0, 0]
lightgrey = [.8, .8, .8]

mydpi = 72.0
f = plt.figure(figsize=(1280.0/mydpi, 720.0/mydpi), dpi=mydpi)
g = gridspec.GridSpec(2, 2)

ax1 = plt.subplot(g[1, 0:2])
colormap = plt.cm.gist_gray
plt.gca().set_prop_cycle(color=[colormap(i) for i in np.linspace(0, 0.5, 3)])
plt.plot(tau, e_hat[:, 0], lw=2, label=r'$\hat{e}_1$')
plt.plot(tau, e_hat[:, 1], lw=2, label=r'$\hat{e}_2$')
plt.plot(tau, e_hat[:, 2], lw=2, label=r'$\hat{e}_3$')
plt.legend()
ax1.set_xlim([tau[0], tau[-1]])
ax1.set_title('First three eigenvectors')
ax1.set_xlabel('time to maturity (yrs)')

ax2 = plt.subplot(g[0, 0])
ax2.bar(omega, lambda2_hat[:k_]/lambda2_hat[0], width=omega[1]-omega[0],
        facecolor=lightgrey, label=r'empirical')
ax2.plot(omega, lambda2_omega/lambda2_omega[0], color=darkred,
         lw=1.5, label=r'theoretical')
ax2.set_ylim([0, 1.1])
plt.legend()
ax2.set_title('Eigenvalues')
ax2.set_xlabel('frequences')

ax3 = plt.subplot(g[0, 1])
ax3.bar(omega, r2_omega[:k_], facecolor=lightgrey, width=omega[1]-omega[0],
        label=r'empirical')
ax3.plot(omega, r2_omega, color=darkred, lw=1.5, label=r'theoretical')
ax3.set_ylim([0, 1.1])
plt.legend()
ax3.set_title('$\mathcal{R}^2$')
ax3.set_xlabel('frequences')

add_logo(f, location=4)
plt.tight_layout()
