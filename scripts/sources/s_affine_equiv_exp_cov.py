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

# # s_affine_equiv_exp_cov [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_affine_equiv_exp_cov&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=exp-cov-ellip).

# ## Prepare the environment

# +
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from scipy import linalg

# rc('text', usetex=True)
# rcParams['text.latex.preamble']=[r"\usepackage{amsmath} \usepackage{amssymb}"]

from arpym.statistics.simulate_normal import simulate_normal
from arpym.tools.plot_ellipse import plot_ellipse
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_affine_equiv_exp_cov-parameters)

m = np.array([0.17, 0.06])  # parameters of lognormal
s2 = np.array([[0.06, -0.03], [-0.03, 0.02]])
a = np.array([-0.5, 0.5])  # parameters of an invertible affine transformation
b = np.array([[-1, -0.1], [0.01, 0.8]])
j_ = 1000  # number of simulations
r = 3  # radius

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_affine_equiv_exp_cov-implementation-step01): Generate the sample of X and of Y

x = np.exp(simulate_normal(m, s2, j_)).T
y = np.tile(a.reshape(2,1), (1, j_)) + b @ x

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_affine_equiv_exp_cov-implementation-step02): Compute expectation and covariance of X and Y

mu_x = np.exp(m + 0.5*np.diag(s2))
sigma2_x = np.diag(mu_x) @ (np.exp(s2)-np.ones([2, 2])) @ np.diag(mu_x)
mu_y = a + b @ mu_x
sigma2_y = b @ sigma2_x @ b.T

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_affine_equiv_exp_cov-implementation-step03): Perform computations for the plots

x_bar = plot_ellipse(mu_x, sigma2_x, color='b', r=r, line_width=4, display_ellipse=False, plot_axes=True, plot_tang_box=True)
y_bar = plot_ellipse(mu_y, sigma2_y, color='r', r=r, line_width=4, display_ellipse=False, plot_axes=True, plot_tang_box=True)

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_affine_equiv_exp_cov-implementation-step04): Select points on exp-cov ellipsoids

x0 = x_bar[1, :]
y0 = a + b @ x0

# ## Plot

# +
# plt.style.use('arpm')
fig = plt.figure(figsize=(1280/72, 720/72), dpi=72)
plt.plot(x.T[:, 0], x.T[:, 1], "o", color='b', markersize=3, label=r'Simulations of $\boldsymbol{X}$')
plt.plot(y.T[:, 0], y.T[:, 1], "o", color='r', markersize=3, label=r'Simulations of $\boldsymbol{Y}$')
plt.plot(x_bar[:, 0], x_bar[:, 1], color='b', linewidth=3, label=r'Exp-cov ellipsoid of $\boldsymbol{X}$')
plt.plot(y_bar[:, 0], y_bar[:, 1], color='r', linewidth=3, label=r'Exp-cov ellipsoid of $\boldsymbol{Y}$')
plt.plot(x0[0], x0[1], 'b', marker='.', markersize=15)
plt.plot(y0[0], y0[1], 'r', marker='.', markersize=15)
plt.xlabel('$X_1$', fontsize=17)
plt.ylabel('$X_2$', fontsize=17)
# plt.legend()
plt.title('Affine transformation of a bivariate lognormal', fontsize=20, fontweight='bold')
# add_logo(fig)
# -







