#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # s_uniform_to_gamma [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_uniform_to_gamma&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-unif-to-gamma).

# +
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from arpym.tools import histogram_sp, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_uniform_to_gamma-parameters)

k = 4  # shape parameter
theta = 4  # scale parameter
j_ = 100000  # number of scenarios

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_uniform_to_gamma-implementation-step01): Generate a uniform sample

u = np.random.rand(j_)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_uniform_to_gamma-implementation-step02): Apply inverse transform sampling

x = stats.gamma.ppf(u, k , theta)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_uniform_to_gamma-implementation-step03): Compute the empirical histogram of the pdf of the new sample

k_bar = np.round(5*np.log(j_))
[f_hist, xi] = histogram_sp(x, k_=k_bar)

# ## Plots

plt.style.use('arpm')
fig = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)
plt.title('Uniform-to-gamma mapping', fontsize=20, fontweight='bold')
# empirical pdf
plt.bar(xi, f_hist, width=xi[1]-xi[0], facecolor=[.7, .7, .7],
        edgecolor='k',  label='empirical pdf')
# analytical pdf
plt.plot(xi, stats.gamma.pdf(xi, k , theta),
         color='red', lw=5, label='gamma pdf')
plt.grid(True)
plt.ylim([0, 1.1*np.max(f_hist)])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=17)
add_logo(fig, location=2, set_fig_size=False)
plt.tight_layout()
