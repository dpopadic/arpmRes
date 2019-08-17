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

# # s_lognorm_to_uniform [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_lognorm_to_uniform&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-lognorm-to-unif).

# +
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from arpym.tools import histogram_sp, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_lognorm_to_uniform-parameters)

mu = 2  # location parameter
sigma2 = 1  # dispersion parameter
j_ = 100000  # number of scenarios

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_lognorm_to_uniform-implementation-step01): Generate a lognormal sample

x = stats.lognorm.rvs(mu, np.sqrt(sigma2), size=j_)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_lognorm_to_uniform-implementation-step02): Apply the lognormal cdf to the sample

u = stats.lognorm.cdf(x, mu, np.sqrt(sigma2))

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_lognorm_to_uniform-implementation-step03): Compute the empirical histogram of the pdf of the grade sample

k_bar = np.round(3*np.log(j_))
[f_hist, xi] = histogram_sp(u, k_=k_bar)

# ## Plots

plt.style.use('arpm')
fig = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)
plt.title('Lognormal-to-uniform mapping', fontsize=20, fontweight='bold')
# empirical pdf
plt.bar(xi, f_hist, width=xi[1]-xi[0], facecolor=[.7, .7, .7],
        edgecolor='k',  label='empirical pdf')
# uniform analytical pdf
plt.plot(np.linspace(0, 1, num=50), np.ones(50),
         color='red', lw=1.5, label='uniform pdf')
plt.grid(True)
plt.ylim([0, 1.25*max(xi)])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=17)
add_logo(fig, location=2, set_fig_size=False)
plt.tight_layout()
