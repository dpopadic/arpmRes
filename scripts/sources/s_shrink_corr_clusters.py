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

# # s_shrink_corr_clusters [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_shrink_corr_clusters&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerMST).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import KMeans

from arpym.statistics import simulate_normal
from arpym.tools import add_logo
# -

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_shrink_corr_clusters-implementation-step00): Load data

path = '../../../databases/global-databases/equities/db_stocks_SP500/'
# stocks values
df_stocks = pd.read_csv(path + 'db_stocks_sp.csv', index_col=0, header=[0, 1])
sectors = np.array(df_stocks.columns.levels[0])  # sector names
labels = np.array(df_stocks.columns.codes)[0, :]  # sector indices

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_shrink_corr_clusters-implementation-step01): Compute the correlation matrix from the log-returns

epsi = np.diff(np.log(df_stocks), axis=0)  # log-returns
c2 = np.corrcoef(epsi.T)  # historical correlation
t_, n_ = epsi.shape

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_shrink_corr_clusters-implementation-step02): Sort the correlation matrix by sectors

i_s = np.argsort(labels)
c2_sec = c2[np.ix_(i_s, i_s)]  # correlation matrix sorted by sectors

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_shrink_corr_clusters-implementation-step03): Compute the initial clustering by computing the average of each sector

x = simulate_normal(np.zeros(n_), c2, 2 * n_)
k_ = sectors.shape[0]  # number of sectors
c0 = np.zeros((2 * n_, k_))
for k in range(k_):
    c0[:, k] = np.mean(x[:, labels == k], axis=1)

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_shrink_corr_clusters-implementation-step04): Determine clusters and sort the correlation matrix accordingly

kmeans = KMeans(n_clusters=k_, init=c0.T, n_init=1).fit(x.T)  # fit
i_c = np.argsort(kmeans.labels_)
c2_clus = c2[np.ix_(i_c, i_c)]

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_shrink_corr_clusters-implementation-step05): Compute the minimum spanning tree

d = np.sqrt(2 * (1 - c2))  # distance matrix
gr = nx.from_numpy_matrix(d)
mst = nx.minimum_spanning_tree(nx.from_numpy_matrix(d))  # min. spanning tree

# ## Plots

# +
plt.style.use('arpm')

sector_names_short = ['CDis', 'CSta', 'Ene', 'Fin', 'HC',
                      'Ind', 'IT', 'Mat', 'TS', 'U']
# color settings
c_sect = np.array([[0.2,  0.6,  0],  # dark green
                   [0.5,  1,    0.5],  # light green
                   [0.8,  0.8,  0.8],  # light gray
                   [0.6,  0.5,  0.1],  # brown
                   [0.27, 0.4,  0.9],  # blue
                   [0,    1,    1],  # light blue
                   [1,    0.5,  1],  # pink
                   [0,    0,    0],  # black
                   [1,    0,    0],  # red
                   [1,    1,    0]])  # yellow

c_max = np.max(c2 - np.eye(n_))
c_min = np.min(c2 - np.eye(n_))

# Correlations
fig, axes = plt.subplots(1, 2)
plt.sca(axes[0])
plt.imshow(c2_sec - np.eye(n_), vmin=c_min, vmax=c_max, aspect='equal')

l_s = np.cumsum(np.bincount(labels[i_s]))
for k in reversed(range(10)):
    plt.plot([l_s[k], l_s[k]], [1, n_], 'r-')
    plt.plot([1, n_], [l_s[k], l_s[k]], 'r-')
    plt.plot([1, l_s[k]], [1, l_s[k]], color=c_sect[k, :], markersize=8)

tick = np.r_[l_s[0] / 2, l_s[:-1] + np.diff(l_s) / 2]
plt.xticks(tick, sector_names_short, rotation=90)
plt.yticks(tick, sector_names_short)
plt.title('Sector Clusters')
plt.grid(False)

plt.sca(axes[1])
plt.imshow(c2_clus - np.eye(n_), vmin=c_min, vmax=c_max, aspect='equal')

l_c = np.cumsum(np.bincount(kmeans.labels_[i_c]))
for k in reversed(range(10)):
    plt.plot([l_c[k], l_c[k]], [1, n_], 'r-')
    plt.plot([1, n_], [l_c[k], l_c[k]], 'r-')
    plt.plot([1, l_c[k]], [1, l_c[k]], color=c_sect[k, :], markersize=8)

plt.title('Correlation Clusters')
plt.grid(False)

add_logo(fig)
plt.tight_layout()

# Minimum spanning trees
fig, ax = plt.subplots(1, 2)
plt.sca(ax[0])
gr = nx.from_numpy_matrix(d)
x = nx.minimum_spanning_tree(gr)
pos = nx.nx_pydot.graphviz_layout(x, prog='neato')
nx.draw_networkx(x, pos=pos, node_shape='.', width=1, node_size=1,
                 node_color='b', ax=ax[0], with_labels=False)
for i in range(k_-1):
    idx = i_s.flatten()[int(l_s[i]):int(l_s[i+1])+1]
    for id in idx:
        plt.plot(pos[id][0], pos[id][1], marker='.', markersize=10,
                 c=c_sect[i, :])
plt.axis('off')
plt.title('Colors by sectors')
plt.sca(ax[1])
nx.draw_networkx(x, pos=pos, node_shape='.', width=1, node_size=1,
                 node_color='b', ax=ax[1], with_labels=False)
plt.axis('off')
for i in range(k_-1):
    idx = i_c[int(l_c[i]):int(l_c[i+1])+1]
    for id in idx:
        plt.plot(pos[id][0], pos[id][1], marker='.', markersize=10,
                 c=c_sect[i, :])
plt.title('Colors by clusters')

add_logo(fig)
plt.tight_layout()
