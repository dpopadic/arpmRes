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

# # s_strong_dominance [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_strong_dominance&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=doc-s_strong_dominance).

# +
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from arpym.statistics import simulate_normal
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_strong_dominance-parameters)

mu_ = np.array([1, 0])  # mean vector of jointly normal variables
sigma2_ = np.array([[1, 0],
                    [0, 1]])  # covariance matrix
j_ = 1000  # number of simulations

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_strong_dominance-implementation-step01): Simulate jointly normal random variables X_1 and X_2

x = simulate_normal(mu_, sigma2_, j_)
x_1, x_2 = x[:, 0], x[:, 1]

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_strong_dominance-implementation-step02): Simulate X_3 = X_2 + Y, Y chi-squared with 1 degree of freedom

x_3 = x_2 + sp.stats.chi2.rvs(1, size=(1, j_))

# ## Plots

# +
# set figure specifications
plt.style.use('arpm')
f, ax = plt.subplots(1, 2, figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0,
                     subplot_kw={'aspect': 'equal'})

# create subplot for general case: x_2 versus x_1
plt.sca(ax[0])
plt.scatter(x_2, x_1, marker='.')
min1 = np.floor(mu_[0]-4*np.sqrt(sigma2_[0, 0]))
min2 = np.floor(mu_[1]-4*np.sqrt(sigma2_[1, 1]))
max1 = np.ceil(mu_[0]+4*np.sqrt(sigma2_[0, 0]))
max2 = np.ceil(mu_[1]+4*np.sqrt(sigma2_[1, 1]))
plt.axis([min(min1, min2), max(max1, max2), min(min1, min2), max(max1, max2)])
plt.plot(np.linspace(min(min1, min2), max(max1, max2)),
         np.linspace(min(min1, min2), max(max1, max2)),
         color='black', lw=2)
plt.title('General case', fontsize=20, fontweight='bold')
plt.xlabel(r'$X_2$', fontsize=17)
plt.ylabel(r'$X_1$', fontsize=17)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)

# create subplot of strong dominance: x_2 versus x_3
plt.sca(ax[1])
plt.scatter(x_2, x_3, marker='.')
plt.axis([min2, max2+4, min2, max2+4])
plt.plot(np.linspace(min2, max2+4),
         np.linspace(min2, max2+4),
         color='black', lw=2)
plt.title('Strong dominance', fontsize=20, fontweight='bold')
plt.xlabel(r'$X_2$', fontsize=17)
plt.ylabel(r'$X_3$', fontsize=17)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)

add_logo(f, location=4, set_fig_size=False)
plt.tight_layout()
