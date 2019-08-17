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

# # s_shrink_cov_glasso [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_shrink_cov_glasso&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=Glasso_estimate).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

from arpym.estimation import cov_2_corr, markov_network
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_shrink_cov_glasso-parameters)

i_ = 60  # number of invariants
lambda_vec = np.arange(0, 0.6, 10**-2)  # glasso penalty
n_plot = 40  # number of stocks for plotting

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_shrink_cov_glasso-implementation-step00): Load data

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

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_shrink_cov_glasso-implementation-step01): Glasso shrinkage

k = int(i_*(i_-1))  # shrink all covariances to 0
sig2_glasso, _, phi2_glasso, lam, conv, _ =\
    markov_network(sig2_hat, k, lambda_vec)

# ## Plots

# +
plt.style.use('arpm')

# Graph
nonzero = np.count_nonzero(phi2_glasso[:n_plot, :n_plot])
num_edge = (nonzero - i_) / 2

fig = plt.figure(figsize=(1280.0/72, 720.0/72), dpi=72)

ax = plt.subplot2grid((2, 4), (0, 1), colspan=2)
bb = np.where(phi2_glasso[:n_plot, :n_plot] != 0, 1, 0)
rows, cols = np.where(bb != 0)
edges = list(zip(rows.tolist(), cols.tolist()))
gr = nx.Graph()
gr.add_edges_from(edges)
nx.draw_circular(gr, node_shape='o', node_color='b', ax=ax)
plt.axis([-1.05, 1.05, -1.05, 1.5])
text1 = 'Optimal penalty = %1.2e' % lam
plt.text(-1, 1.25, text1, verticalalignment='bottom',
         horizontalalignment='left', fontsize=20)
text2 = 'Num. edges = %3.0f' % num_edge
plt.text(-1, 1.1, text2, verticalalignment='bottom',
         horizontalalignment='left', fontsize=20)
plt.title('Markov network structure', fontweight='bold', fontsize=20)

# Covariances
minncov = np.min(np.c_[sig2_hat[:n_plot, :n_plot],
                       sig2_glasso[:n_plot, :n_plot]])
maxxcov = np.max(np.c_[sig2_hat[:n_plot, :n_plot],
                       sig2_glasso[:n_plot, :n_plot]])
minncorr = np.min(np.c_[phi2_hat[:n_plot, :n_plot],
                        phi2_glasso[:n_plot, :n_plot]])
maxxcorr = np.max(np.c_[phi2_hat[:n_plot, :n_plot],
                        phi2_glasso[:n_plot, :n_plot]])

ax1 = plt.subplot2grid((2, 4), (1, 0), colspan=1)
ax1 = sns.heatmap(sig2_hat[:n_plot, :n_plot],
                  cmap='BrBG',
                  center=0,
                  xticklabels=stocks_names[:n_plot],
                  yticklabels=stocks_names[:n_plot],
                  vmin=minncov,
                  vmax=maxxcov,
                  square=True)
plt.title('HFP corr.', fontweight='bold', fontsize=20)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)

ax12 = plt.subplot2grid((2, 4), (1, 1), colspan=1)
ax12 = sns.heatmap(phi2_hat[:n_plot, :n_plot],
                   cmap='BrBG',
                   center=0,
                   xticklabels=stocks_names[:n_plot],
                   yticklabels=stocks_names[:n_plot],
                   vmin=minncorr,
                   vmax=maxxcorr,
                   square=True)
plt.title('HFP inv. corr.', fontweight='bold', fontsize=20)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)

ax2 = plt.subplot2grid((2, 4), (1, 2), colspan=1)
ax2 = sns.heatmap(sig2_glasso[:n_plot, :n_plot],
                  cmap='BrBG',
                  center=0,
                  xticklabels=stocks_names[:n_plot],
                  yticklabels=stocks_names[:n_plot],
                  vmin=minncov,
                  vmax=maxxcov,
                  square=True)
plt.title('Glasso corr.', fontweight='bold', fontsize=20)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)

ax22 = plt.subplot2grid((2, 4), (1, 3), colspan=1)
ax22 = sns.heatmap(phi2_glasso[:n_plot, :n_plot],
                   cmap='BrBG',
                   center=0,
                   xticklabels=stocks_names[:n_plot],
                   yticklabels=stocks_names[:n_plot],
                   vmin=minncorr,
                   vmax=maxxcorr,
                   square=True)
plt.title('Glasso inv. corr.', fontweight='bold', fontsize=20)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)

add_logo(fig, axis=ax, set_fig_size=False)
plt.tight_layout()
