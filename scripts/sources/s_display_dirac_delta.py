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

# # s_display_dirac_delta [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_display_dirac_delta&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerDeltaApprox).

# +
import numpy as np
import matplotlib.pyplot as plt

from arpym.statistics import gaussian_kernel
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_display_dirac_delta-parameters)

y = np.array([0, 0])
h2 = 0.01
k_ = 200

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_display_dirac_delta-implementation-step01): Compute the Gaussian kernel

# +
x_1_grid = np.linspace(-2+y[0], 2+y[0], k_)
x_2_grid = np.linspace(-2+y[1], 2+y[1], k_)
x_1, x_2 = np.meshgrid(x_1_grid, x_2_grid)

delta_h2_y_x = np.array([gaussian_kernel(h2, y, x)
                         for x in zip(np.ravel(x_1),
                                      np.ravel(x_2))]).reshape(x_1.shape)
# -

# ## Plots

# +
plt.style.use('arpm')
fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
ax.view_init(30, 30)

ax.contour(x_1, x_2, delta_h2_y_x, 50, colors='blue')

ax.text(y[0], y[1], np.max(delta_h2_y_x)*1.1,
        r'$\delta_{\mathbf{%.2f}}^{(\mathbf{y})}(\mathbf{x})$' % h2,
        color="blue", fontsize=17)

ax.scatter(y[0], y[1], 0, s=30, color='k')
ax.text(y[0]+0.1, y[1]+0.1, 0, r'$\mathbf{y}$', fontsize=17, color='k')

plt.xlabel(r'$x_1$', labelpad=15, fontsize=17)
plt.ylabel(r'$x_2$', labelpad=15, fontsize=17)
plt.title('Approximation of Dirac delta with Gaussian kernel')

tick_step = 2
ticklabels = [ax.xaxis.get_ticklabels(), ax.yaxis.get_ticklabels()]
for tl in ticklabels:
    for n, label in enumerate(tl):
        if n % tick_step != 0:
            label.set_visible(False)

ax.set_zticks([])
ax.grid(False)

add_logo(fig)
