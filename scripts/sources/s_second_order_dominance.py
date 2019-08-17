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

# # s_second_order_dominance [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_second_order_dominance&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=doc-s_second_order_dominance).

# +
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_second_order_dominance-parameters)

mu_1 = 0.2
sigma_1 = np.sqrt(0.1)
mu_2 = 0
sigma_2 = np.sqrt(0.3)

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_second_order_dominance-implementation-step01): Get points on the cdfs to plot

# +
# define x values for calculations
llim = -1
ulim = int(round(max(np.exp(mu_1+3*sigma_1), np.exp(mu_2+3*sigma_2))))
n_grid = 601
x_grid = np.linspace(llim, ulim, n_grid)

# find cdfs for X_1 and X_2
cdf_1 = sp.stats.lognorm.cdf(x_grid, sigma_1, scale=np.exp(mu_1))
cdf_2 = sp.stats.lognorm.cdf(x_grid, sigma_2, scale=np.exp(mu_2))
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_second_order_dominance-implementation-step02): Calculate integrated cdfs

# +
# initialize output arrays
cdf_integral_1 = np.zeros(n_grid)
cdf_integral_2 = np.zeros(n_grid)

# define the functions to be integrated (lognormal cdfs)
cdf_fun_1 = lambda x: sp.stats.lognorm.cdf(x, sigma_1, scale=np.exp(mu_1))
cdf_fun_2 = lambda x: sp.stats.lognorm.cdf(x, sigma_2, scale=np.exp(mu_2))

# calculate the integral of the cdf for each point in x_grid
for n in range(n_grid):
    cdf_integral_1[n] = sp.integrate.quad(cdf_fun_1, -np.Inf, x_grid[n])[0]
    cdf_integral_2[n] = sp.integrate.quad(cdf_fun_2, -np.Inf, x_grid[n])[0]
# -

# ## Plots

# +
# set figure specifications
plt.style.use('arpm')
f, ax = plt.subplots(1, 2, figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)

# cdf comparison
plt.sca(ax[0])
ax[0].set_xlim([0, 1])
ax[0].set_ylim([llim, ulim])
plt.plot(cdf_1, x_grid, lw=2, color='C0',
         label=r'$F_{X_{1}}(x)$')
plt.plot(cdf_2, x_grid, lw=2, color='C3',
         label=r'$F_{X_{2}}(x)$')
plt.title('cdf comparison', fontsize=20, fontweight='bold')
plt.xlabel('cdf', fontsize=17)
plt.ylabel(r'$x$', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.legend(fontsize=17, borderpad=0.5, labelspacing=0.5)

# cdf integral comparison
plt.sca(ax[1])
ax[1].set_xlim([0, np.ceil(max(cdf_integral_1[-1], cdf_integral_2[-1]))])
ax[1].set_ylim([llim, ulim])
plt.plot(cdf_integral_1, x_grid, lw=2, color='C0',
         label=r'$\int_{-\infty}^{x} F_{X_{1}}(s)ds$')
plt.plot(cdf_integral_2, x_grid, lw=2, color='C3',
         label=r'$\int_{-\infty}^{x} F_{X_{2}}(s)ds$')
plt.title('cdf integrals comparison', fontsize=20, fontweight='bold')
plt.xlabel('cdf integrals', fontsize=17)
plt.ylabel(r'$x$', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.legend(fontsize=17, borderpad=0.5, labelspacing=0.5)

plt.tight_layout()
add_logo(f, location=4, set_fig_size=False)
