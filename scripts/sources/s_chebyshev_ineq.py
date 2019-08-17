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

# # s_chebyshev_ineq [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_chebyshev_ineq&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ChMahalSimul).

# +
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams

rc('text', usetex=True)
rcParams['text.latex.preamble']=[r"\usepackage{amsmath} \usepackage{amssymb}"]

from arpym.statistics import simulate_normal
from arpym.tools import plot_ellipse, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_chebyshev_ineq-parameters)

mu = np.array([1, 1])  # expectation
rho = 0.6  # correlation
sigma2 = np.array([[1, rho], [rho, 1]])  # covariance
m = np.array([1, 1])  # volume of generic ellipsoid
theta = np.pi/3  # rotation angle
r_theta = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])  # rotation matrix
r = 2  # radius of ellipsoids
j_ = 5000  # number of scenarios

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_chebyshev_ineq-implementation-step01): Compute the square dispersion of the generic ellipsoid via rotation

s2 = r_theta @ sigma2 @ r_theta.T

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_chebyshev_ineq-implementation-step02): Perform computations for the plot

x = simulate_normal(mu, sigma2, j_)
x_ec = plot_ellipse(mu, sigma2, r=r, display_ellipse=False, plot_axes=True, plot_tang_box=True,
             color='orange')
x_g = plot_ellipse(m, s2, r=r, display_ellipse=False, plot_axes=True, plot_tang_box=True,
             color='b')


# ## Plots

# +
plt.style.use('arpm')
orange = [.9, .4, 0]
grey = [.5, .5, .5]

fig = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)
plt.axis('equal')
plt.plot([], [], color=orange, lw=2)  # dummy plot for legend
plt.plot([], [], color='b', lw=2)  # dummy plot for legend
plt.plot(x[:, 0], x[:, 1], "o", color=grey, markersize=3)
plt.plot(x_ec[:, 0], x_ec[:, 1], color=orange, linewidth=2)
plt.plot(x_g[:, 0], x_g[:, 1], color='b', linewidth=2)
plt.legend(('Exp-cov ellipsoid',
           'Generic ellipsoid (same volume)'), loc=2)
plt.title(r"Chebyshev's inequality", fontsize=20, fontweight='bold')
plt.xlabel('$X_1$', fontsize=17)
plt.ylabel('$X_2$', fontsize=17)
add_logo(fig)
