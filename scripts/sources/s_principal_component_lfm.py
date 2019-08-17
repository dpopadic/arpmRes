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

# # s_principal_component_lfm [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_principal_component_lfm&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-lfmhid-cor-copy-1).

# +
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from arpym.statistics import simulate_normal
from arpym.tools import pca_cov, plot_ellipsoid, transpose_square_root, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_principal_component_lfm-parameters)

mu_x = np.array([1., 0., 3.])  # expectation of the target variable
sig2_x = np.array([[1., -0.4, 0.68],
                 [-0.4, 1., -0.58],
                 [0.68, -0.58, 1.]])  # covariance of the target variable
sig2 = np.eye(3)  # scale matrix
n_ = len(mu_x)  # target dimension
k_ = 2  # number of factors
j_ = 1000  # number of scenarios

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_principal_component_lfm-implementation-step00): Compute Riccati root of the scale matrix

sig = transpose_square_root(sig2)

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_principal_component_lfm-implementation-step01): Compute the eigenvalues and eigenvectors

sig_inv = np.linalg.solve(sig, np.eye(n_))
e, lambda2 = pca_cov(sig_inv@sig2_x@sig_inv)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_principal_component_lfm-implementation-step02): Compute the loadings, factor-construction matrix and r-square

# +
beta = sig @ e[:, :k_]  # principle-component loadings
gamma = e[:, :k_].T@sig_inv  # factor-construction matrix
alpha = mu_x  # optimal coefficient a

r2_sig2 = np.sum(lambda2[:k_]) / np.sum(lambda2)  # r-squared
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_principal_component_lfm-implementation-step03): Compute mean and cov of factors and factor-recovered variables

# +
mu_z = 0
sig2_z = np.diag(lambda2[:k_])

mu_x_pc_bar = mu_x
betagamma = beta @ gamma
sig2_x_pc_bar = betagamma @ sig2_x @ betagamma.T

m = np.r_[np.eye(n_) - beta @ gamma, gamma]
sig2_uz = m @ sig2_x @ m.T  # joint covariance of U and Z
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_principal_component_lfm-implementation-step04): Generate target variable and factor-recovered simulations

# +
x = simulate_normal(mu_x, sig2_x, j_)
x_pc_bar = mu_x + (x - mu_x)@betagamma.T

x_rs = (x - mu_x)@sig_inv.T  # rescaled/shifted target
x_pc_rs = (x_pc_bar - mu_x)@sig_inv.T  # rescaled/shifted predicted
# -

# ## Plots

# +
plt.style.use('arpm')

scale = 4

# compute principal axis and plane
pcrange = np.arange(-scale*1.5, scale*1.5+scale*0.5, scale*0.5)
r_size = len(pcrange)

plane = np.zeros((n_, r_size, r_size))
for r1 in range(r_size):
    for r2 in range(r_size):
        plane[:, r1, r2] = e[:, 0] * np.sqrt(lambda2[0]) * pcrange[r1] + \
                           e[:, 1] * np.sqrt(lambda2[1]) * pcrange[r2]

prange = np.arange(0, scale+scale*0.5, scale*0.5)
r_size = len(prange)

e1 = np.zeros((n_, r_size))
e2 = np.zeros((n_, r_size))
e3 = np.zeros((n_, r_size))
for r in range(r_size):
    e1[:, r] = e[:, 0] * np.sqrt(lambda2[0]) * prange[r]
    e2[:, r] = e[:, 1] * np.sqrt(lambda2[1]) * prange[r]
    e3[:, r] = e[:, 2] * np.sqrt(lambda2[2]) * prange[r]


fig1, ax1 = plot_ellipsoid(np.zeros(3),
                          sig_inv@sig2_x@sig_inv,
                          r=scale,
                          plot_axes=False,
                          ellip_color=(.8, .8, .8),
                          ellip_alpha=0.3,
                          n_points=0)

# plot plane
ax1.view_init(30, -140)
ax1.plot_surface(plane[0], plane[1], plane[2],
                   color=[.8, .8, .8], shade=False, alpha=0.2)
h00 = Line2D([0], [0], linestyle="none", c=[.8, .8, .8],
             marker='o', label='Principal component plane')
# plot eigenvectors
h01 = ax1.plot(e1[0], e1[1], e1[2], color='r', lw=2, label='Principal axes')
ax1.plot(e2[0], e2[1], e2[2], color='r', lw=2)
ax1.plot(e3[0], e3[1], e3[2], color='r', lw=2)

# rescaled random sample
h02 = ax1.plot(x_rs[:, 0], x_rs[:, 1], x_rs[:, 2], '.',
                 color='b', markersize=3, label='Target variables')
ax1.grid(False)
ax1.set_xlabel(r'$X_{1}$')
ax1.set_ylabel(r'$X_{2}$')
ax1.set_zlabel(r'$X_{3}$')
ax1.legend(handles=[h00, h01[0], h02[0]])

add_logo(fig1, size_frac_x=1/8)


fig2, ax2 = plot_ellipsoid(np.zeros(3),
                          sig_inv@sig2_x@sig_inv,
                          r=scale,
                          plot_axes=False,
                          ellip_color=(.8, .8, .8),
                          ellip_alpha=0.3,
                          n_points=0)

# plot plane
ax2.view_init(30, -140)
ax2.plot_surface(plane[0], plane[1], plane[2],
                   color=[.8, .8, .8], shade=False, alpha=0.2)
h00 = Line2D([0], [0], linestyle="none", c=[.8, .8, .8],
             marker='o', label='Principal component plane')
# plot eigenvectors
h01 = ax2.plot(e1[0], e1[1], e1[2], color='r', lw=2, label='Principal axes')
ax2.plot(e2[0], e2[1], e2[2], color='r', lw=2)
ax2.plot(e3[0], e3[1], e3[2], color='r', lw=2)
# rescaled projected sample
h02 = ax2.plot(x_pc_rs[:, 0], x_pc_rs[:, 1], x_pc_rs[:, 2], '.',
                 markersize=3, color='g', label='Prediction')
ax2.grid(False)
ax2.set_xlabel(r'$X_{1}$')
ax2.set_ylabel(r'$X_{2}$')
ax2.set_zlabel(r'$X_{3}$')
ax2.legend(handles=[h00, h01[0], h02[0]])

add_logo(fig2, size_frac_x=1/8)
