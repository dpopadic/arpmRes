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

# # s_different_length_series [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_different_length_series&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=exer-diff-length-copy-1).

# +
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from arpym.estimation import exp_decay_fp, fit_locdisp_mlfp,\
                             fit_locdisp_mlfp_difflength
from arpym.tools import plot_ellipse, colormap_fp, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_different_length_series-parameters)

nu = 4  # degrees of freedom in MLFP estimation
tau_hl = 2*252  # half life decay parameter for flexible probabilities
trunc = 0.8  # proportion of the time series to be dropped
tol = 10 ** -6  # MLFP routine convergence threshold

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_different_length_series-implementation-step00): Upload data

# +
times_to_maturity = np.round_(np.array([1, 2, 3, 5, 7, 8, 10]), 2)
path = '../../../databases/global-databases/fixed-income/db_yields/data.csv'
y_db = pd.read_csv(path, parse_dates=['dates'], skip_blank_lines=True)

y = y_db[times_to_maturity.astype(float).astype(str)].values
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_different_length_series-implementation-step01): Compute the swap rates daily changes

# daily changes
epsi = np.diff(y, 1, axis=0)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_different_length_series-implementation-step02): Flexible probabilities

p = exp_decay_fp(len(epsi), tau_hl)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_different_length_series-implementation-step03): Maximum likelihood with flexible probabilities - complete series

mu, s2 = fit_locdisp_mlfp(epsi, p=p, nu=nu, threshold=tol)

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_different_length_series-implementation-step04): Drop the first portion of the observations from the 2yr and 5yr series

r = int(np.floor(len(epsi)*trunc))
epsi_dl = epsi.copy()
epsi_dl[:r, [1, 3]] = np.nan  # drop observations

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_different_length_series-implementation-step05): Maximum likelihood with flexible probabilities - different length

mu_dl, s2_dl = fit_locdisp_mlfp_difflength(epsi_dl, p=p, nu=nu, threshold=tol)

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_different_length_series-implementation-step06): Maximum likelihood with flexible probabilities - truncated series

# +
epsi_trunc = epsi[r:, :]  # truncated time series
p_trunc = p[r:] / np.sum(p[r:])  # flexible probabilities

# MLFP estimation
mu_trunc, s2_trunc = fit_locdisp_mlfp(epsi_trunc, p=p_trunc, nu=nu, threshold=tol)
# -

# ## Plots

# +
plt.style.use('arpm')

# scatter colormap and colors
cm, c = colormap_fp(p, grey_range=np.arange(0.25, 0.91, 0.01), c_min=0,
                    c_max=1, value_range=[1, 0])

# Scatter plot
epsi_25 = epsi[:, [1, 3]]  # select invariants

fig = plt.figure()
ax = plt.subplot2grid((4, 1), (0, 0), rowspan=3)

plt.scatter(epsi_25[:r, 0], epsi_25[:r, 1], 20, marker='o', linewidths=3,
            edgecolor=[.9, .7, .7], facecolor='none')  # Dropped obs.
plt.axis('equal')
plt.scatter(epsi_25[:, 0], epsi_25[:, 1], 20, c=c, marker='o', cmap=cm)
plt.axis([np.percentile(epsi_25[:, 0], 5), np.percentile(epsi_25[:, 0], 95),
          np.percentile(epsi_25[:, 1], 5), np.percentile(epsi_25[:, 1], 95)])
plt.xlabel('2yr rate daily changes')
plt.ylabel('5yr rate daily changes')
plt.ticklabel_format(style='sci', scilimits=(0, 0))

# Ellipsoids
mu_25 = mu[[1, 3]]  # select invariants expectations
mu_dl_25 = mu_dl[[1, 3]]
mu_trunc_25 = mu_trunc[[1, 3]]
s2_25 = s2[np.ix_([1, 3], [1, 3])]  # select invariants covariance
s2_dl_25 = s2_dl[np.ix_([1, 3], [1, 3])]
s2_trunc_25 = s2_trunc[np.ix_([1, 3], [1, 3])]

ell = plot_ellipse(mu_25, s2_25, color='b')
ell1 = plot_ellipse(mu_dl_25, s2_dl_25, color='tomato')
ell2 = plot_ellipse(mu_trunc_25, s2_trunc_25, color='g')

# legend
leg = plt.legend(['MLFP - complete series', 'MLFP - different len',
                  'MLFP - truncated series', 'Dropped observations'])

# bottom plot: highlight missing observations in the dataset as white spots
ax1 = plt.subplot2grid((4, 1), (3, 0))
plot_dates = np.array(y_db.dates)
na = np.ones(epsi.T.shape)
# na=1: not-available data (2y and 5y series are placed as last two entries)
na[-2:, :r] = 0
plt.imshow(na, aspect='auto')
plt.ylim([epsi.shape[1], 0])
ax1.set_xticks([])
ax1.set_yticks([5, 6])
ax1.set_yticklabels([' 2yr', ' 5yr'])
plt.grid(False)
add_logo(fig, axis=ax)
plt.tight_layout()
