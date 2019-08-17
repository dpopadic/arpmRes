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

# # s_simulate_unif_in_ellipse [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_simulate_unif_in_ellipse&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-unif-inside-radial-mcex).

# +
import numpy as np
import matplotlib.pyplot as plt

from arpym.tools import plot_ellipse, add_logo
from arpym.statistics import simulate_unif_in_ellips
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_simulate_unif_in_ellipse-parameters)

mu = np.array([4, 1])  # location of the ellipsoid
sigma2 = np.array([[3, 1.5], [1.5, 1]])  # dispersion of the ellipsoid
j_ = 1000  # number of scenarios

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_simulate_unif_in_ellipse-implementation-step01): Generate scenarios

x, r, y = simulate_unif_in_ellips(mu, sigma2, j_)
ry = r * y

# ## Plots

# +
plt.style.use('arpm')
fig = plt.figure()

# Unit circle
unitcircle = plot_ellipse(np.zeros(2), np.eye(2), color='b', line_width=0.5)

# Ellipse(mu, sigma2)
ellipse, ellpoints, *_ = plot_ellipse(mu, sigma2, color='r', line_width=0.5)

# Plot scenarios of the uniform component Y
ply = plt.plot(y[:, 0], y[:, 1],
               markersize=3, markerfacecolor='b', marker='o', linestyle='none',
               label='$\mathbf{Y}$: uniform on the unit circle')


# Plot scenarios of the component RY
plry = plt.plot(ry[:, 0], ry[:, 1],
                markersize=3, markerfacecolor='g', marker='o',
                linestyle='none',
                label='$\mathbf{RY}$: uniform inside the unit circle')

# Plot scenarios of X
plx = plt.plot(x[:, 0], x[:, 1],
               markersize=3, markerfacecolor='r', marker='o', linestyle='none',
               label='$\mathbf{X}$: uniform inside an ellipse')

plt.legend()
plt.axis('equal')

add_logo(fig)
