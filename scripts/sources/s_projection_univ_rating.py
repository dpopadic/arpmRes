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

# # s_projection_univ_rating [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_projection_univ_rating&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-trans-prob-ep).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patch
from matplotlib.collections import PatchCollection

from arpym.statistics import project_trans_matrix, simulate_markov_chain_univ
from arpym.tools import histogram_sp, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_projection_univ_rating-parameters)

# +
x_tnow = np.array([3])  # initial rating
deltat = 120 # time to horizon (in months)
m_ = 120  # number of monitoring times
j_ = 1000  # number of scenarios
# -

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_projection_univ_rating-implementation-step00): Upload data

# +
# upload database generated from s_fit_discrete_markov_chain
path = '../../../databases/temporary-databases/'
df_p = pd.read_csv(path + 'db_trans_matrix.csv', index_col=0)
p = np.array(df_p).reshape(8, 8)  # yearly transition matrix
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_projection_univ_rating-implementation-step01): Compute probability mass function at the horizon conditioned on the current rating

# +
# compute projected transition matrix
p_dt = project_trans_matrix(p, m_/12)
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_projection_univ_rating-implementation-step02): Compute Monte Carlo scenarios

# +
x_tnow_thor = simulate_markov_chain_univ(x_tnow, p, (deltat/m_)*np.ones(m_)/12, j_)
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_projection_univ_rating-implementation-step03): Compute histogram

# +
# exact conditional ratings distribution
pdf_thor = p_dt[x_tnow, :]
# empirical conditional ratings distribution
bins = np.arange(8)
f, xi = histogram_sp(x_tnow_thor[:, -1], p=None, xi=bins)
# -

# ## Plots

# +
# settings
plt.style.use('arpm')
scale = 30
fig, ax = plt.subplots(1, 1)
plt.axis([0, m_ + np.max(f) * scale + 26, 0, 9])
rat_col = list([[0, 166 / 255, 0], [75 / 255, 209 / 255, 29 / 255],
                [131 / 255, 213 / 255, 32 / 255],
                [188 / 255, 217 / 255, 34 / 255],
                [221 / 255, 195 / 255, 36 / 255],
                [225 / 255, 144 / 255, 38 / 255],
                [229 / 255, 92 / 255, 40 / 255],
                [233 / 255, 42 / 255, 47 / 255]])
c1 = [0.8, 0.8, 0.8]  # light grey
c2 = [0.2, 0.2, 0.2]  # dark grey
j_sel = 100

# paths
for j in range(j_sel):
    plt.plot(np.arange(m_+1), 1+x_tnow_thor[j, :].flatten(), color=c1)
plt.xticks(np.linspace(0, m_, 5), fontsize=17)
plt.yticks(np.arange(1, 9), ['AAA', 'AA', 'A', 'BBB', 'BB',
                             'B', 'CCC', 'D'], fontsize=17,
           fontweight='bold')
for ytick, color in zip(ax.get_yticklabels(), rat_col):
    ytick.set_color(color)
plt.title('Projection of Markov chain', fontsize=20)
plt.ylabel('Rating', fontsize=17)
plt.xlabel('Time (months)', fontsize=17)
plt.text(m_ + np.max(f) * scale + 0.1, 8.3, 'Rating probs.', fontsize=17)
plt.text(m_ + np.max(f) * scale + 5.1, 0.9, '0', fontsize=14)
plt.text(m_ + np.max(f) * scale + 5.1, 7.9, '1', fontsize=14)

# histogram and rating probabilities bar
plt.plot([m_, m_], [0.2, 8.8], color=c2)
r1 = []
r2 = []
vert_y = np.r_[0, 7 * np.cumsum(pdf_thor)] + 1
height = np.diff(vert_y)
for s in np.arange(0, 8):
    rect1 = patch.Rectangle((m_, s + 0.75), f[s] * scale, 0.5)
    rect2 = patch.Rectangle((m_ + np.max(f) * scale + 12, vert_y[s]),
                            1.0, height[s])
    r1.append(rect1)
    r2.append(rect2)
pc1 = PatchCollection(r1, facecolor=rat_col, alpha=1)
pc2 = PatchCollection(r2, facecolor=rat_col, edgecolor='k', alpha=1)
ax.add_collection(pc1)
ax.add_collection(pc2)

add_logo(fig)
plt.tight_layout()
