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

# # s_rand_ols_estim [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_rand_ols_estim&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExFactBayesOLSEstim).

# +
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from arpym.estimation import fit_lfm_ols
from arpym.tools import histogram_sp, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_rand_ols_estim-parameters)

t_ = 10  # len of the time series
j_ = 1000  # number of simulations
b = 1  # real value of b
sigma2 = 4  # real value of sigma

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_rand_ols_estim-implementation-step01): Generate simulations of factor, conditional residual and randomized time series

z = np.random.randn(t_)
u = np.sqrt(sigma2) * np.random.randn(j_, t_)
x = b * z + u

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_rand_ols_estim-implementation-step02): Compute simulations of the least squares estimators

_, b_hat, sigma2_u, _ = fit_lfm_ols(x.T, z, fit_intercept=False)
s_hat = np.diag(sigma2_u)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_rand_ols_estim-implementation-step03): Compute the empirical and analytical pdfs of OLS estimator of loading

f_b_emp, b_grid = histogram_sp(b_hat)
f_b_ana = sp.stats.norm.pdf(b_grid, b, np.sqrt(sigma2 / t_))

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_rand_ols_estim-implementation-step04): Compute the empirical and analytical pdfs of OLS estimator of dispersion

f_s_emp, s_grid = histogram_sp(s_hat)
f_s_ana = sp.stats.wishart.pdf(s_grid, t_ - 1, sigma2 / t_)

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_rand_ols_estim-implementation-step05): Compute then analytical joint pdfs

f_bs_ana = np.outer(f_b_ana, f_s_ana)

# ## Plots

# +
plt.style.use('arpm')

# residual pdf
fig1 = plt.figure()
u_grid = np.linspace(-4 * np.sqrt(sigma2), 4 * np.sqrt(sigma2), 200)
f_u_ana = sp.stats.norm.pdf(u_grid, 0, np.sqrt(sigma2))
plt.plot(u_grid, f_u_ana, color=[0.25, 0.25, 1])
plt.title('Distribution of conditional residual')
plt.xlabel(r'$U_t|z_t,\beta,\sigma^2$')

add_logo(fig1, location=1)
plt.tight_layout()

# loading pdf
fig2 = plt.figure()
plt.bar(b_grid, f_b_emp, width=b_grid[1]-b_grid[0], facecolor=[0.8, 0.8, 0.8])
plt.plot(b_grid, f_b_ana, color=[0.25, 0.25, 1], lw=1.5)
plt.title('OLS loading distribution')
plt.legend(['empirical pdf', 'analytical pdf'])
plt.xlabel(r'$Loadings|\beta,\sigma^2$')

add_logo(fig2, location=2)
plt.tight_layout()

# dispersion pdf
fig3 = plt.figure()
plt.bar(s_grid, f_s_emp, width=s_grid[1]-s_grid[0], facecolor=[0.8, 0.8, 0.8])
plt.plot(s_grid, f_s_ana, color=[0.25, 0.25, 1], lw=1.5)
plt.title('OLS dispersion distribution')
plt.legend(['empirical pdf', 'analytical pdf'])
plt.xlabel(r'$Dispersion|\beta,\sigma^2$')

add_logo(fig3, location=2)
plt.tight_layout()

# joint distribution
fig4 = plt.figure()
plt.plot(b_hat, s_hat, '*', markersize=4, color=[0.5, 0.5, 0.5])
plt.contour(b_grid, s_grid, f_bs_ana.T, 6)
plt.plot([], [], 'k', lw=1.3)
plt.legend(['empirical scatter plot', 'analytical contour lines'])
plt.xlabel(r'Loadings|$\beta,\sigma^2$')
plt.ylabel(r'Dispersion|$\beta,\sigma^2$')
plt.title('Joint distribution of OLS estimators')

add_logo(fig4, location=4)
plt.tight_layout()
