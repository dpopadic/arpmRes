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

# # s_regression_lfm [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_regression_lfm&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-lfmtime-cor-copy-1).

# +
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from arpym.estimation import cov_2_corr
from arpym.statistics import simulate_normal
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_regression_lfm-parameters)

mu_xz = np.array([3., 3., 3.])  # joint expectation
sigma2_xz = np.array([[1., 0.21, 0.35],
                      [0.21, 4., 0.6],
                      [0.35, 0.6, 1.]])  # joint covariance
j_ = 1000  # number of scenarios

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_regression_lfm-implementation-step01): Compute optimal loadings

# +
n_ = 1
k_ = 2

sigma_xz = sigma2_xz[:n_, n_:]
sigma2_z = sigma2_xz[n_:, n_:]
mu_z = mu_xz[n_:]
mu_x = mu_xz[:n_]
beta = sigma_xz@np.linalg.inv(sigma2_z)
alpha = mu_x - beta @ mu_z
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_regression_lfm-implementation-step02): Compute expectation and covariance of prediction

mu_xreg_bar = alpha + beta@mu_z
sigma2_xreg_bar = beta @ sigma2_z @ beta.T

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_regression_lfm-implementation-step03): Compute the r-squared

# +
c2_xz, _ = cov_2_corr(sigma2_xz)

sigma2_x = sigma2_xz[:n_, :n_]
r2 = np.trace(sigma_xz@np.linalg.inv(sigma2_z)@sigma_xz.T)/np.trace(sigma2_x)
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_regression_lfm-implementation-step04): Compute joint distribution of residulas and factors

a = np.zeros(n_ + k_)
a[:n_] = -alpha
b = np.eye(n_ + k_)
b[:n_, n_:] = -beta
mu_uz = a + b @ mu_xz
sigma2_uz = b @ sigma2_xz @ b.T

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_regression_lfm-implementation-step05): Compute simulations of target variable and factors

xz = simulate_normal(mu_xz, sigma2_xz, j_)
x_reg = alpha + beta @ xz[:, n_:].T

# ## Plots

# +
# number of simulations to plot
d = 200

z_1_low = np.percentile(xz[:, n_], 1)
z_1_upp = np.percentile(xz[:, n_], 99)
z_1 = np.arange(z_1_low, z_1_upp, 0.5)
z_2_low = np.percentile(xz[:, n_+1], 1)
z_2_upp = np.percentile(xz[:, n_+1], 99)
z_2 = np.arange(z_2_low, z_2_upp, 0.5)

[z_1, z_2] = np.meshgrid(z_1, z_2)
x_reg_plane = alpha + beta[0, 0] * z_1 + beta[0, 1] * z_2


x_max = np.max(np.r_[xz[:d, 0], x_reg_plane.reshape(-1)])
x_min = np.min(np.r_[xz[:d, 0], x_reg_plane.reshape(-1)])

z1_min = np.min(z_1)
z1_max = np.max(z_1)

z2_min = np.min(z_2)
z2_max = np.max(z_2)

lim_max = np.max([x_max, z1_max, z2_max])
lim_min = np.min([x_min, z1_min, z2_min])

plt.style.use('arpm')

fig = plt.figure()

ax2 = fig.add_subplot(121, projection='3d')

ax2.grid(b=False)
ax2.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax2.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax2.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

sur1 = ax2.plot_wireframe(z_1, z_2, x_reg_plane,
                          edgecolor=[220/250, 220/250, 220/250])
sct1 = ax2.scatter(xz[:d, n_], xz[:d, n_ + 1], xz[:d, 0], marker='.',
                   color='b')

ax2.set_zlim([lim_min, lim_max])
ax2.set_xlim([lim_min, lim_max])
ax2.set_ylim([lim_min, lim_max])
ax2.set_xlabel('$Z_1$')
ax2.set_ylabel('$Z_2$')
ax2.set_zlabel('$X$')

ax2.view_init(10, ax2.azim)

ax3 = fig.add_subplot(122, projection='3d')

ax3.grid(b=False)
ax3.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax3.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax3.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))


sur2 = ax3.plot_wireframe(z_1, z_2, x_reg_plane,
                          edgecolor=[220/250, 220/250, 220/250])
sct2 = ax3.scatter(xz[:d, n_], xz[:d, n_ + 1], x_reg[0, :d]+0.1, marker='.',
                   color='g', alpha=1)

ax3.set_zlim([lim_min, lim_max])
ax3.set_xlim([lim_min, lim_max])
ax3.set_ylim([lim_min, lim_max])
ax3.set_xlabel('$Z_1$')
ax3.set_ylabel('$Z_2$')
ax3.set_zlabel('$X$')

ax3.view_init(10, ax3.azim)

dummy_legend_lines = [Line2D([0], [0], marker='o', markerfacecolor='b',
                             color='w', lw=4, markersize=8),
                      Line2D([0], [0], marker='o', markerfacecolor='g',
                             color='w', lw=4, markersize=8),
                      Line2D([0], [0], marker="s",
                             markerfacecolor=[220/250, 220/250, 220/250],
                             lw=4, color='w', markersize=8)]

plt.legend(dummy_legend_lines,
           ['Scenarios', 'Predicted scenarios', 'Regression plane'])

add_logo(fig, size_frac_x=1/8)
plt.tight_layout()
