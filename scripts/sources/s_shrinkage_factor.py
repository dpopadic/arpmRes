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

# # s_shrinkage_factor [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_shrinkage_factor&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerLRD).

# +
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from arpym.estimation import exp_decay_fp, cov_2_corr, factor_analysis_paf
from arpym.statistics import meancov_sp
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_shrinkage_factor-parameters)

tau_hl = 180  # half life
k_ = 25  # number of hidden factors

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_shrinkage_factor-implementation-step00): Load data

path = '../../../databases/temporary-databases/'
xi = np.array(pd.read_csv(path + 'db_GARCH_residuals.csv', index_col=0))
t_, n_ = xi.shape

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_shrinkage_factor-implementation-step01): Compute the HFP correlation

p = exp_decay_fp(t_, tau_hl)
_, sigma2 = meancov_sp(xi, p)
c2, _ = cov_2_corr(sigma2)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_shrinkage_factor-implementation-step02): Compute the loadings and residual variances via PAF factor analysis

beta, delta2 = factor_analysis_paf(c2, k_)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_shrinkage_factor-implementation-step03): Compute the factor analysis correlation and the Frobenius norm

c2_paf = beta @ beta.T + np.diag(delta2)
d_fro = np.linalg.norm(c2 - c2_paf, ord='fro') / \
        np.linalg.norm(c2, ord='fro') * 100.

# ## Plots

# +
plt.style.use('arpm')

cmax = 0.75
bmax = 0.5
bmin = -0.5
cbar = np.linspace(0, cmax, 6)
bbar = np.linspace(bmin, bmax, 6)

fig, ax = plt.subplots(2, 2)

plt.sca(ax[0, 0])
cax_1 = plt.imshow(abs(c2_paf), vmin=0, vmax=cmax, aspect='equal')
cbar_1 = fig.colorbar(cax_1, ticks=cbar, format='%.2f', shrink=0.53)
cbar_1.ax.set_yticklabels(['0', '0.15', '0.3', '0.45', '0.6', '>0.75'])
plt.grid(False)
plt.title('Factor analysis correlation (abs)')

plt.sca(ax[0, 1])
cax_2 = plt.imshow(abs(c2), vmin=0, vmax=cmax, aspect='equal')
cbar_2 = fig.colorbar(cax_2, ticks=cbar, format='%.2f', shrink=0.53)
cbar_2.ax.set_yticklabels(['0', '0.15', '0.3', '0.45', '0.6', '>0.75'])
plt.grid(False)
plt.title('Correlation (abs)')

plt.sca(ax[1, 0])
cax_1 = plt.imshow(beta, vmin=bmin, vmax=bmax, aspect='equal')
cbar_1 = fig.colorbar(cax_1, ticks=bbar, format='%.2f', shrink=0.53)
cbar_1.ax.set_yticklabels(['<-0.5', '-0.3', '-0.1', '0.1', '0.3', '>0.5'])
plt.grid(False)
plt.title('Loadings')
plt.text(-0.8, -0.2, 'Frobenius percentage distance:  %2.1f' % d_fro, transform=plt.gca().transAxes)
plt.text(-0.8, -0.3, 'Low - rank dimension: k = %2i' % k_, transform=plt.gca().transAxes)

plt.sca(ax[1, 1])
cax_2 = plt.imshow(abs(np.diag(delta2)), vmin=0, vmax=cmax, aspect='equal')
cbar_2 = fig.colorbar(cax_2, ticks=cbar, format='%.2f', shrink=0.53)
cbar_2.ax.set_yticklabels(['0', '0.15', '0.3', '0.45', '0.6', '>0.75'])
plt.grid(False)
plt.title('Residual variances')
x_pos = -100
y_pos = 60
# plt.text(x_pos, y_pos, 'Frobenius percentage distance:  %2.1f' % d_fro)
# plt.text(0.5, 0.5, 'Low - rank dimension: k = %2i' % k_, transform=plt.gca().transAxes)
add_logo(fig, size_frac_x=1/8, location=1)
plt.tight_layout()
