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

# # s_full_dependence [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_full_dependence&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-2-full-co-dep).

# +
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma

from arpym.tools import histogram_sp, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_full_dependence-parameters)

# +
j_ = 10 ** 4  # number of simulations
k1 = 1  # shape parameter of gamma distrubution
k2 = 5  # shape parameter of gamma distrubution
theta1 = 0.8  # scale parameter of gamma distribution
theta2 = 1.3  # scale parameter of gamma distribution
# -

# ## Step1: Generate a uniform sample

# +
u = np.random.random(j_)
# -

# ## Step2: Compute the marginal (Gamma) simulations

# +
gamma1 = gamma(k1, scale=theta1)
gamma2 = gamma(k2, scale=theta2)
x1 = gamma1.ppf(u)
x2 = gamma2.ppf(u)
# -

# ## Step3: Compute the normalized histograms of marginal simulations

# +
f_x1, ksi_x1 = histogram_sp(x1)
f_x2, ksi_x2 = histogram_sp(x2)
# -

# ## Plots

# +
plt.style.use('arpm')
fig = plt.figure()
# colors
teal = [0.2344, 0.582, 0.5664]
light_grey = [0.6, 0.6, 0.6]
#
x1_min = min(x1)
x1_max = max(x1)
x2_min = min(x2)
x2_max = max(x2)
x1_grid = np.arange(x1_min, x1_max + 0.01, 0.01)
x2_grid = np.arange(x2_min, x2_max + 0.01, 0.01)
#
ax1 = plt.subplot2grid((64, 80), (0, 17), colspan=47, rowspan=47)
ax1.scatter(x1, x2, marker='o', label='scatter plot of $(X_1, X_2)^{\prime}$', color=light_grey)
ax1.tick_params(axis='x', bottom=True, top=False, labelcolor='none')
ax1.tick_params(axis='y', which='major', pad=-20, left=True, right=False, labelcolor='none')
ax1.set_xlabel(r'$X_1$', fontdict={'size': 16}, labelpad=-40)
ax1.set_ylabel(r'$X_2$', fontdict={'size': 16}, labelpad=-30)
#
ax2 = plt.subplot2grid((64, 80), (50, 17), colspan=47, rowspan=14, sharex=ax1)
ax2.tick_params(axis='x', top=True, labeltop=True, bottom=False, labelbottom=False, pad=0)
ax2.tick_params(axis='y', which='major', direction='out', pad=0)
ax2.invert_yaxis()
ax2.bar(ksi_x1, f_x1, ksi_x1[1]-ksi_x1[0], facecolor=teal, label='marginal histogram')
ax2.plot(x1_grid, gamma1.pdf(x1_grid), color='k')
#
ax3 = plt.subplot2grid((64, 80), (0, 0), colspan=14, rowspan=47, sharey=ax1)
ax3.tick_params(axis='y', left=False, right=True, labelleft=False, labelright=True, rotation=90, pad=5)
ax3.tick_params(axis='x', which='major')
ax3.invert_xaxis()
ax3.plot(gamma2.pdf(x2_grid), x2_grid, color='k', label=' marginal pdf')
ax3.barh(ksi_x2, f_x2, ksi_x2[1]-ksi_x2[0], facecolor=teal)
#
fig.legend(loc=[0.75, 0.5], edgecolor='white', labelspacing=1)
add_logo(fig, axis=ax1, set_fig_size=True, location=4)
# -
