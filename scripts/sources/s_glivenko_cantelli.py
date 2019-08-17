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

# # s_glivenko_cantelli [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_glivenko_cantelli&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerGCplot).

# +
import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt

from arpym.tools import histogram_sp, add_logo
from arpym.statistics import cdf_sp
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_glivenko_cantelli-parameters)

t_ = 2500  # number of observations
mu = 0  # location parameter of the lognormal distribution
sigma2 = 0.25  # scale parameter of the lognormal distribution
n_ = 500  # number of grid points for the cdf

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_glivenko_cantelli-implementation-step01): Generate lognormal sample

epsi = lognorm.rvs(sigma2, scale=np.exp(mu), size=t_)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_glivenko_cantelli-implementation-step02): Compute the historical pdf

p = np.ones(t_)/t_  # uniform probabilities
pdf_hist_eps, xi = histogram_sp(epsi, p=p, k_=10*np.log(t_))

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_glivenko_cantelli-implementation-step03): Compute the historical cdf

x_grid = np.linspace(0, np.max(epsi), n_+1)
cdf_hist_eps = cdf_sp(x_grid, epsi, p)

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_glivenko_cantelli-implementation-step04): Compute the true lognormal pdf and cdf

pdf_eps = lognorm.pdf(x_grid, sigma2, scale=np.exp(mu))
cdf_eps = lognorm.cdf(x_grid, sigma2, scale=np.exp(mu))

# ## Plots

# +
plt.style.use('arpm')

# Display the historical pdf and overlay the true pdf

gr = [0.4, 0.4, 0.4]  # colors settings

fig, ax = plt.subplots(2, 1, figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)
plt.sca(ax[0])
emp2 = plt.bar(xi, pdf_hist_eps, width=xi[1]-xi[0],
               facecolor=gr, edgecolor='k')
plt.plot(x_grid, pdf_eps, color='b', lw=1.5)
plt.xlim([np.min(x_grid), np.max(x_grid)])
plt.ylim([0, max(pdf_hist_eps) + 0.1])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Historical PDF', fontsize=20, fontweight='bold')

# Display the historical cdf and overlay the true cdf

plt.sca(ax[1])
plt.plot(x_grid, cdf_eps, color='b', lw=1)
emp = plt.plot(x_grid, cdf_hist_eps, color=gr, lw=1.5)
plt.title('Historical CDF', fontsize=20, fontweight='bold')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim([0, max(x_grid)])
plt.ylim([-0.001, 1.001])
plt.legend(['True', 'Historical'], fontsize=17)
add_logo(fig, set_fig_size=False)
plt.tight_layout()
