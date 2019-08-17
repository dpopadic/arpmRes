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

# # s_bayesian_estimation [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_bayesian_estimation&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_bayesian_estimation).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from arpym.estimation import cov_2_corr
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_bayesian_estimation-parameters)

i_ = 60  # number of invariants
pri_t_ = 20  # t_pri/t = nu_pri/t_ = pri_t_
n_plot = 50  # number of stocks for plotting

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_bayesian_estimation-implementation-step00): Load data

# +
path_temp = '../../../databases/temporary-databases/'

# Invariants
db_epsi = pd.read_csv(path_temp + 'db_fit_garch_stocks_epsi.csv',
                      index_col=0, parse_dates=True)
db_epsi = db_epsi.iloc[:, :i_]

dates = db_epsi.index
t_ = len(dates)
stocks_names = db_epsi.columns
epsi = db_epsi.values

# Location-dispersion
db_locdisp = pd.read_csv(path_temp + 'db_fit_garch_stocks_locdisp.csv')
mu_hat = db_locdisp.loc[:, 'mu_hat'].values[:i_]
sig2_hat = db_locdisp.loc[:, 'sig2_hat'].values
i_tot = int(np.sqrt(len(sig2_hat)))
sig2_hat = sig2_hat.reshape(i_tot, i_tot)[:i_, :i_]

sig2_hat = cov_2_corr(sig2_hat)[0]
phi2_hat = np.linalg.solve(sig2_hat, np.eye(i_))
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_bayesian_estimation-implementation-step01): Bayesian estimation

# +
# Prior
mu_pri = np.zeros(i_)
sig2_pri = np.diag(cov_2_corr(sig2_hat)[1]**2)
t_pri = int(pri_t_*t_)
nu_pri = int(pri_t_*t_)

# Posterior
t_pos = t_pri + t_
nu_pos = nu_pri + t_

mu_pos = t_pri/t_pos*mu_pri + t_/t_pos*mu_hat
mu_diff = np.atleast_2d(mu_pri-mu_hat).T
sig2_pos = nu_pri/nu_pos*sig2_pri +\
            t_/nu_pos*sig2_hat +\
            1/(nu_pos*(1/t_ + 1/t_pri))*mu_diff@mu_diff.T
c2_pos = cov_2_corr(sig2_pos)[0]
phi2_pos = np.linalg.solve(c2_pos, np.eye(i_))
# -

# ## Plots

# +
plt.style.use('arpm')

# Graph
zero = 10**-2
phi2_pos[np.abs(phi2_pos) < zero] = 0
nonzero = np.count_nonzero(phi2_pos)
num_edge = (nonzero - i_) / 2

fig = plt.figure(figsize=(1280.0/72, 720.0/72), dpi=72)

ax = plt.subplot2grid((2, 4), (0, 1), colspan=2)
bb = np.where(phi2_pos[:n_plot, :n_plot] != 0, 1, 0)
rows, cols = np.where(bb != 0)
edges = list(zip(rows.tolist(), cols.tolist()))
gr = nx.Graph()
gr.add_edges_from(edges)
nx.draw_circular(gr, node_shape='o', node_color='b', ax=ax)
plt.axis([-1.05, 1.05, -1.05, 1.5])
text1 = 'Threshold = %1.2e' % zero
plt.text(-1, 1.25, text1, verticalalignment='bottom',
         horizontalalignment='left', fontsize=20)
text2 = 'Num. edges = %3.0f' % num_edge
plt.text(-1, 1.1, text2, verticalalignment='bottom',
         horizontalalignment='left', fontsize=20)
plt.title('Markov network structure', fontweight='bold', fontsize=20)

# Covariances
minncov = np.min(np.c_[sig2_hat[:n_plot, :n_plot],
                       sig2_pos[:n_plot, :n_plot]])
maxxcov = np.max(np.c_[sig2_hat[:n_plot, :n_plot],
                       sig2_pos[:n_plot, :n_plot]])
minncorr = np.min(np.c_[phi2_hat[:n_plot, :n_plot],
                        phi2_pos[:n_plot, :n_plot]])
maxxcorr = np.max(np.c_[phi2_hat[:n_plot, :n_plot],
                        phi2_pos[:n_plot, :n_plot]])

ax1 = plt.subplot2grid((2, 4), (1, 0), colspan=1)
ax1 = sns.heatmap(sig2_hat[:n_plot, :n_plot],
                  cmap='BrBG',
                  center=0,
                  xticklabels=stocks_names[:n_plot],
                  yticklabels=stocks_names[:n_plot],
                  vmin=minncov,
                  vmax=maxxcov,
                  square=True)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.title('HFP corr.',
          fontweight='bold', fontsize=20)
ax12 = plt.subplot2grid((2, 4), (1, 1), colspan=1)
ax12 = sns.heatmap(phi2_hat[:n_plot, :n_plot],
                   cmap='BrBG',
                   center=0,
                   xticklabels=stocks_names[:n_plot],
                   yticklabels=stocks_names[:n_plot],
                   vmin=minncorr,
                   vmax=maxxcorr,
                   square=True)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.title('HFP inv. corr.',
          fontweight='bold', fontsize=20)
ax2 = plt.subplot2grid((2, 4), (1, 2), colspan=1)
ax2 = sns.heatmap(sig2_pos[:n_plot, :n_plot],
                  cmap='BrBG',
                  center=0,
                  xticklabels=stocks_names[:n_plot],
                  yticklabels=stocks_names[:n_plot],
                  vmin=minncov,
                  vmax=maxxcov,
                  square=True)

plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.title('Bayes posterior corr.\n$t_{pri} = \\nu_{pri} = %d*t\_$' % pri_t_,
          fontweight='bold', fontsize=20)

ax22 = plt.subplot2grid((2, 4), (1, 3), colspan=1)
ax22 = sns.heatmap(phi2_pos[:n_plot, :n_plot],
                   cmap='BrBG',
                   center=0,
                   xticklabels=stocks_names[:n_plot],
                   yticklabels=stocks_names[:n_plot],
                   vmin=minncorr,
                   vmax=maxxcorr,
                   square=True)

plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.title('Bayes posterior inv. corr.\n$t_{pri} = \\nu_{pri} = %d*t\_$' % pri_t_,
          fontweight='bold', fontsize=20)

add_logo(fig, axis=ax, set_fig_size=False, size_frac_x=1/12)
plt.tight_layout()
