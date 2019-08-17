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

# # s_bivariate_wishart [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_bivariate_wishart&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExWishartBivariate).

# +
import numpy as np
import matplotlib.pyplot as plt

from arpym.statistics import meancov_wishart, meancov_inverse_wishart
from arpym.statistics import simulate_wishart
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_bivariate_wishart-parameters)

nu = 6
sig_1 = 1
sig_2 = 1
rho_12 = 0
a = np.array([-3, 2])
j_ = 1000

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_bivariate_wishart-implementation-step01): Generate Monte Carlo simulations of the bivariate Wishart random matrix

# +
sig2 = np.array([[sig_1**2, rho_12*sig_1*sig_2],
                 [rho_12*sig_1*sig_2, sig_2**2]])

w2 = simulate_wishart(nu, sig2, j_)
w_11 = w2[:, 0, 0]
w_12 = w2[:, 0, 1]
w_22 = w2[:, 1, 1]
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_bivariate_wishart-implementation-step02): Compute the expectation and the covariance of the Wishart distribution

e_w2, cv_w2 = meancov_wishart(nu, sig2)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_bivariate_wishart-implementation-step03): Compute the dispersion parameter of the transformed variable

sig2_a = a.T@sig2@a

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_bivariate_wishart-implementation-step04): Compute the expectation and the covariance of the inverse-Wishart distribution

# +
n_ = sig2.shape[0]
psi2 = np.linalg.solve(sig2, np.eye(n_))

e_sig2, cv_sig2 = meancov_inverse_wishart(nu, psi2)
# -

# ## Plots

# +
range_scale = [5, 95]
refine = 70

low_11 = np.percentile(w_11, range_scale[0])
high_11 = np.percentile(w_11, range_scale[1])
range_w_11 = np.linspace(low_11, high_11, refine+1)

low_12 = np.percentile(w_12, range_scale[0])
high_12 = np.percentile(w_12, range_scale[1])
range_w_12 = np.linspace(low_12, high_12, refine+1)

low_22 = np.percentile(w_22, range_scale[0])
high_22 = np.percentile(w_22, range_scale[1])
range_w_22 = np.linspace(low_22, high_22, refine+1)

plt.style.use('arpm')
fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
w_1_1, w_1_2 = np.meshgrid(range_w_11, range_w_12)

# surface w_11 * w_22 - w_12**2 = 0
w_2_2 = w_1_2 ** 2 / w_1_1
ax.plot_surface(w_1_1, w_1_2, w_2_2, color='lightgray', shade=False, zorder=1)

# plane w_11 + w_22 = 0
w_22_tr = -w_1_1
ax.plot_surface(w_1_1, w_1_2, w_22_tr, color='gray', shade=False)

# bivariate Wishart distribution

indices = [j for j in range(j_)
           if range_w_11[0] < w_11[j] < range_w_11[-1]
           if range_w_12[0] < w_12[j] < range_w_12[-1]
           if range_w_22[0] < w_22[j] < range_w_22[-1]]

ax.plot(w_11[indices], w_12[indices], w_22[indices], '.', zorder=2)
ax.set_xlim([range_w_11[0], range_w_11[-1]])
ax.set_ylim([range_w_12[0], range_w_12[-1]])
ax.set_zlim([range_w_22[0]-20, range_w_22[-1] + 20])

ax.set_xlabel(r'$[\mathbf{w}]_{1,1}$')
ax.set_ylabel(r'$[\mathbf{w}]_{1,2}$')
ax.set_zlabel(r'$[\mathbf{w}]_{2,2}$')

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.text(range_w_11[-1]-10, range_w_12[-1], range_w_22[-1],
        r'$[\mathbf{w}]_{1,1}[\mathbf{w}]_{2,2}-[\mathbf{w}]_{2,2}^2 = 0$',
        color="black")
ax.text(range_w_11[0], range_w_12[0], -range_w_11[-1]-5,
        r'$[\mathbf{w}]_{1,1}+[\mathbf{w}]_{2,2}= 0$', color="black")

add_logo(fig)
