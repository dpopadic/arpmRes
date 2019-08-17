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

# # s_ellipsoid_multiv_exp_cov [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_ellipsoid_multiv_exp_cov&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=doc-s_ellipsoid_multiv_loc_disp).

# +
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams

rc('text', usetex=True)
rcParams['text.latex.preamble']=[r"\usepackage{amsmath} \usepackage{amssymb}"]

from arpym.tools import add_logo, pca_cov
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_ellipsoid_multiv_exp_cov-parameters)

mu = np.array([1, 1])  # expectation vector
sigma2 = np.array([[1, 0.7],
                  [0.7, 1]])  # covariance matrix

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_ellipsoid_multiv_exp_cov-implementation-step01): Generate points on the unit sphere

theta = np.linspace(0, 2*np.pi, num=200)
y = np.array([[np.cos(angle), np.sin(angle)] for angle in theta]).T

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_ellipsoid_multiv_exp_cov-implementation-step02): Calculate spectral decomposition

e, lambda2_vec = pca_cov(sigma2)
e[[1, 0]] = e[[0, 1]]
diag_lambda = np.diag(np.sqrt(lambda2_vec))

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_ellipsoid_multiv_exp_cov-implementation-step03): Stretch the unit circle: multiply by eigenvalues

z = np.matmul(diag_lambda, y)

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_ellipsoid_multiv_exp_cov-implementation-step04): Rotate the ellipsoid: multiply by eigenvectors

u = np.matmul(e, z)

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_ellipsoid_multiv_exp_cov-implementation-step05): Translate the ellipsoid: add expectation vector

x = (u.T + mu).T

# ## Plots

# +
# set figure specifications
plt.style.use('arpm')
f = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)
# axis limits
bound = np.max(diag_lambda + np.linalg.norm(mu))
# plot unit circle
ax1 = f.add_subplot(2, 2, 1, aspect='equal')
ax1.set_xlim([-bound, bound])
ax1.set_ylim([-bound, bound])
# show x=1 and y=1
plt.hlines(0, -bound, bound)
plt.vlines(0, -bound, bound)
# turn off axis lines
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.set_title('Unit circle',
              fontsize=17, fontweight='bold')
plt.plot(y[0], y[1], label=r'$||\boldsymbol{y}||=1$')
plt.legend(fontsize=14, loc='lower left')

# plot unit circle after multiplication by diag(lambda
ax2 = f.add_subplot(2, 2, 2, aspect='equal')
ax2.set_xlim([-bound, bound])
ax2.set_ylim([-bound, bound])
# show x=1 and y=1
plt.hlines(0, -bound, bound)
plt.vlines(0, -bound, bound)
# turn off axis lines
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.set_title('Stretch: multiplication by eigenvalues',
              fontsize=17, fontweight='bold')
plt.plot(z[0], z[1],
         label=r'$\boldsymbol{z}=Diag(\boldsymbol{\lambda})\times \boldsymbol{y}$')
plt.legend(fontsize=14, loc='lower left')

# plot stretched unit circle after multiplication by eigenvectors
ax3 = f.add_subplot(2, 2, 3, aspect='equal')
ax3.set_xlim([-bound, bound])
ax3.set_ylim([-bound, bound])
# show x=1 and y=1
plt.hlines(0, -bound, bound)
plt.vlines(0, -bound, bound)
# turn off axis lines
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.set_title('Rotation: multiplication by eigenvectors',
              fontsize=17, fontweight='bold')
plt.plot(u[0], u[1], label=r'$\boldsymbol{u} = \boldsymbol{e} \times \boldsymbol{z}$')
plt.legend(fontsize=14, loc='lower left')

# plot stretched and rotated unit circle after addition of location
ax4 = f.add_subplot(2, 2, 4, aspect='equal')
ax4.set_xlim([-bound, bound])
ax4.set_ylim([-bound, bound])
# show x=1 and y=1
plt.hlines(0, -bound, bound)
plt.vlines(0, -bound, bound)
# turn off axis lines
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['bottom'].set_visible(False)
ax4.spines['left'].set_visible(False)
ax4.set_title('Translation: addition of expectation vector',
              fontsize=17, fontweight='bold')
plt.plot(x[0], x[1], label=r'$\boldsymbol{x} = \mathbb{E}\{\boldsymbol{X}\} + \boldsymbol{u}$')
plt.legend(fontsize=14, loc='lower left')

plt.tight_layout()
add_logo(f, location=4, set_fig_size=False)
# -


