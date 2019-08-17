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

# # s_glivenko_cantelli_hfp [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_glivenko_cantelli_hfp&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerGlivCantFP).

# +
import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt

from arpym.tools import histogram_sp, add_logo
from arpym.statistics import cdf_sp
from arpym.estimation import exp_decay_fp, effective_num_scenarios
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_glivenko_cantelli_hfp-parameters)

t_ = 2500
k = 1  # shape parameter for gamma distribution
theta = 2  # scale parameter for gamma distribution
t_star = t_  # target time
tau_hl = t_star / 2  # half-life
n_ = 500

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_glivenko_cantelli_hfp-implementation-step01): Generate a sample from the gamma distribution

epsi = gamma.rvs(k, scale=theta, size=t_)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_glivenko_cantelli_hfp-implementation-step02): Compute the time exponential decay probabilities

p = exp_decay_fp(t_, tau_hl, t_star)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_glivenko_cantelli_hfp-implementation-step03): Compute the effective number of scenarios

ens = effective_num_scenarios(p)

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_glivenko_cantelli_hfp-implementation-step04): Compute the HFP pdf

pdf_hfp_eps, xi = histogram_sp(epsi, p=p, k_=10*np.log(t_))

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_glivenko_cantelli_hfp-implementation-step05): Compute the HFP cdf

x_grid = np.linspace(0, np.max(epsi), n_+1)
cdf_hfp_eps = cdf_sp(x_grid, epsi, p)

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_glivenko_cantelli_hfp-implementation-step06): Compute the true pdf and cdf

pdf_eps = gamma.pdf(x_grid, k, scale=theta)
cdf_eps = gamma.cdf(x_grid, k, scale=theta)

# ## Plots

# +
plt.style.use('arpm')
fig, ax = plt.subplots(2, 1, figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)
gr = [0.4, 0.4, 0.4]  # colors settings

# display the HFP pdf and overlay the true pdf
plt.sca(ax[0])
emp2 = plt.bar(xi, pdf_hfp_eps, width=xi[1]-xi[0],
               facecolor=gr, edgecolor='k')
plt.plot(x_grid, pdf_eps, color='b', lw=1.5)
plt.xlim([np.min(x_grid), np.max(x_grid)])
plt.ylim([0, max(pdf_hfp_eps) + 0.1])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.annotate('ens: '+str(int(round(ens))), xy=(0.93, 0.93),
             xycoords='axes fraction', fontsize=17,
             bbox={'fill': False, 'edgecolor': 'lightgray'})
plt.title('HFP pdf', fontsize=20, fontweight='bold')

# display the HFP cdf and overlay the true cdf
plt.sca(ax[1])
plt.plot(x_grid, cdf_eps, color='b', lw=1)
emp = plt.plot(x_grid, cdf_hfp_eps, color=gr, lw=1.5)
plt.title('HFP cdf', fontsize=20, fontweight='bold')
plt.xlim([0, max(x_grid)])
plt.ylim([-0.001, 1.001])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(['True', 'HFP'], fontsize=17)
add_logo(fig, set_fig_size=False)
plt.tight_layout()
