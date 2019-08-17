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

# # s_saddle_point_vs_mcfp_quadn [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_saddle_point_vs_mcfp_quadn&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-mcfpvs-sp).

# +
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from arpym.statistics import saddle_point_quadn, simulate_quadn, quantile_sp
from arpym.tools import histogram_sp, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_saddle_point_vs_mcfp_quadn-parameters)

n_ = 2
j_ = 100000

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_saddle_point_vs_mcfp_quadn-implementation-step00): Generate random parameters for QuadN dustribution

alpha = np.random.uniform(0, 1)
beta = np.random.uniform(0, 1, size=n_)
gamma = np.random.uniform(0, 1, size=(n_, n_))
gamma = (gamma + gamma.T)/2  # make gamma symmetric and positive (semi)definite
mu = np.random.uniform(0, 1, size=n_)
sigma = np.random.uniform(0, 1, size=(n_, n_))
sigma2 = sigma@sigma.T  # make sigma2 positive definite

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_saddle_point_vs_mcfp_quadn-implementation-step01): Generate quadratic-normal scenarios

y, p_ = simulate_quadn(alpha, beta, gamma, mu, sigma2, j_)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_saddle_point_vs_mcfp_quadn-implementation-step02): Compute the saddle point approximation of the pdf

y_grid = np.linspace(quantile_sp(0.001, y, p_), quantile_sp(0.999, y, p_), 500)
cdf, pdf = saddle_point_quadn(y_grid, alpha, beta, gamma, mu, sigma2)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_saddle_point_vs_mcfp_quadn-implementation-step03): Compute Calculate the heights and bin centers of the normalized empirical histogram

f_hat, grid = histogram_sp(y, p=p_, k_=200)

# ## Plots

# +
plt.style.use('arpm')
darkred = [.9, 0, 0]
lightgrey = [.8, .8, .8]
plt.figure()
mydpi = 72.0
f = plt.figure(figsize=(1280.0/mydpi, 720.0/mydpi), dpi=mydpi)

gs = gridspec.GridSpec(2, 2)
gs.update(wspace=0.5, hspace=0.5)
ax1 = plt.subplot(gs[0, :])
ax1.bar(grid, f_hat, width=grid[1] - grid[0], color=lightgrey, label='Monte Carlo')
ax1.plot(y_grid, pdf, color=darkred, label='Saddle point')
plt.legend()
xlim = [max(grid[0], y_grid[0]), min(grid[-1], y_grid[-1])]
ax1.set_xlim(xlim)
ax1.set_title('Quadratic-normal pdf')

ax2 = plt.subplot(gs[1, :])
ax2.hist(y, bins=15*int(np.log(j_)), density=True,
           color=lightgrey, cumulative=True, label='Monte Carlo')
ax2.plot(y_grid, cdf, color=darkred, label='Saddle point')
plt.legend()
ax2.set_xlim(xlim)
ax2.set_title('Quadratic-normal cdf')

add_logo(f, location=4)
