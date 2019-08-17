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

# # s_weak_dominance [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_weak_dominance&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=doc-s_weak_dominance).

# +
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from arpym.statistics import simulate_normal
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_weak_dominance-parameters)

mu_ = np.array([1, 0])  # mean vector of jointly normal variables
sigma2_ = np.array([[1, 0],
                    [0, 1]])  # covariance matrix
j_ = 5000  # number of simulations

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_weak_dominance-implementation-step01): Calculate marginal cdfs and pdfs

# get pdf and cdf of X_1 and X_2
llim = np.floor(
        min(mu_[0]-5*np.sqrt(sigma2_[0, 0]), mu_[1]-5*np.sqrt(sigma2_[1, 1]))
        )
ulim = np.ceil(
        max(mu_[0]+5*np.sqrt(sigma2_[0, 0]), mu_[1]+5*np.sqrt(sigma2_[1, 1]))
        )
x_grid = np.linspace(llim, ulim, 100)
pdf_1 = sp.stats.norm.pdf(x_grid, mu_[0], np.sqrt(sigma2_[0, 0]))
pdf_2 = sp.stats.norm.pdf(x_grid, mu_[1], np.sqrt(sigma2_[1, 1]))
cdf_1 = sp.stats.norm.cdf(x_grid, mu_[0], np.sqrt(sigma2_[0, 0]))
cdf_2 = sp.stats.norm.cdf(x_grid, mu_[1], np.sqrt(sigma2_[1, 1]))

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_weak_dominance-implementation-step02): Simulate values from X_1 and apply cdfs

# +
# simulate scenarios from X_1
x = simulate_normal(mu_, sigma2_, j_)
x_1 = x[:, 0]

# apply marginal cdfs to the samples
cdf1_x1 = sp.stats.norm.cdf(x_1, mu_[0], sigma2_[0, 0])
cdf2_x1 = sp.stats.norm.cdf(x_1, mu_[1], sigma2_[1, 1])
# -

# ## Plots

# +
# set figure specifications
plt.style.use('arpm')
f, ax = plt.subplots(1, 2, figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)

# pdf comparison
plt.sca(ax[0])
plt.plot(pdf_1, x_grid, lw=2, color='C0', label=r'$f_{X_{1}}(x)$')
plt.plot(pdf_2, x_grid, lw=2, color='C3', label=r'$f_{X_{2}}(x)$')
plt.xlabel('pdf', fontsize=17)
plt.ylabel(r'$x$', fontsize=15, rotation='horizontal')
plt.title('pdf comparison', fontsize=20, fontweight='bold')
plt.legend(fontsize=17, borderpad=0.5, labelspacing=0.5)
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)

# cdf/quantile comparison
plt.sca(ax[1])
plt.plot(cdf_1, x_grid, lw=2, color='C0', label=r'$F_{X_{1}}(x)$')
plt.plot(cdf_2, x_grid, lw=2, color='C3', label=r'$F_{X_{2}}(x)$')
plt.xlabel('cdf', fontsize=17)
plt.title('cdf/quantile comparison', fontsize=20, fontweight='bold')
plt.legend(fontsize=17, borderpad=0.5, labelspacing=0.5)
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)

add_logo(f, location=4, set_fig_size=False)
plt.tight_layout()
plt.show()
plt.close(f)

# weak dominance in terms of strong dominance

# set figure specifications
g = plt.figure(1, figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)
ax_scatter = plt.axes([0.225, 0.305, 0.65, 0.65])
ax_histx = plt.axes([0.225, 0.1, 0.65, 0.2])
ax_histy = plt.axes([0.1, 0.305, 0.12, 0.65])

# scatter plot of cdf1_x1 vs cdf2_x1
ax_scatter.scatter(cdf1_x1[:200], cdf2_x1[:200], marker='.',
                   label=r'cdf transforms applied to sample $\{x_{1}^{(j)}\}_{j=1}^{\bar{j}}\sim X_{1}$')
ax_scatter.plot(range(2), range(2), lw=2, color='black')
ax_scatter.legend(loc='upper left', fontsize=17, borderpad=0.5)
ax_scatter.set_xticklabels([])
ax_scatter.set_yticklabels([])
ax_scatter.spines['top'].set_visible(False)
ax_scatter.spines['right'].set_visible(False)

# histogram of cdf1_x1
ax_histx.hist(cdf1_x1, bins=50, density=True, color='lightgray')
ax_histx.set_xlabel(r'$F_{X_{1}}(X_{1}) \sim U[0,1]$', fontsize=17)
ax_histx.tick_params(axis='x', which='major', labelsize=14)
ax_histx.set_yticklabels([])

# histogram of cdf2_x1
ax_histy.hist(cdf2_x1, bins=50, density=True, color='lightgray',
              orientation='horizontal')
ax_histy.set_ylabel(r'$F_{X_{2}}(X_{1}) \nsim U[0,1]$', fontsize=17)
ax_histy.set_xticklabels([])
ax_histy.tick_params(axis='y', which='major', labelsize=14)

add_logo(g, axis=ax_scatter, location=4, set_fig_size=False)
