#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # s_projection_multiv_ratings [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_projection_multiv_ratings&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-proj-multi-rating-migrations).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arpym.statistics import simulate_markov_chain_multiv, project_trans_matrix
from arpym.tools import histogram2d_sp, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_projection_multiv_ratings-parameters)

x_tnow = np.array([3, 5])  # initial ratings
m_ = 120  # time to horizon (months)
j_ = 1000  # number of scenarios

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_projection_multiv_ratings-implementation-step00): Upload data

# upload database generated from s_fit_discrete_markov_chain
path = '../../../databases/temporary-databases/'
df_p = pd.read_csv(path + 'db_trans_matrix.csv', index_col=0)
p = np.array(df_p).reshape(8, 8)
df_cop = pd.read_csv(path+'db_copula_ratings.csv', index_col=0)
nu = df_cop.nu.values[0]
rho2 = df_cop.rho2.values[0]
rho2 = np.array([[1, rho2], [rho2, 1]])

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_projection_multiv_ratings-implementation-step01): Compute monthly transition matrix

p = project_trans_matrix(p, 1/12)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_projection_multiv_ratings-implementation-step02): Compute Monte Carlo scenarios

x_tnow_thor = simulate_markov_chain_multiv(x_tnow, p, m_, rho2=rho2, nu=nu, j_=j_)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_projection_multiv_ratings-implementation-step03): Compute scenario-probability distribution at the horizon

bins = np.tile(np.arange(0, 8), (2, 1)).T
f, x1, x2 = histogram2d_sp(x_tnow_thor[:, -1, :].squeeze(), xi=bins)

# ## Plots

# +
plt.style.use('arpm')


rat_col = list([[0, 166 / 255, 0], [75 / 255, 209 / 255, 29 / 255],
                [131 / 255, 213 / 255, 32 / 255],
                [188 / 255, 217 / 255, 34 / 255],
                [221 / 255, 195 / 255, 36 / 255],
                [225 / 255, 144 / 255, 38 / 255],
                [229 / 255, 92 / 255, 40 / 255],
                [233 / 255, 42 / 255, 47 / 255]])
c1 = [0.8, 0.8, 0.8]  # light grey
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors = [c1] * 64
colors[x_tnow[1] * 8 + x_tnow[0]] = 'red'
xpos, ypos = np.meshgrid(x1, x2)
xpos = xpos.flatten('F')
ypos = ypos.flatten('F')
zpos = np.zeros_like(xpos)
dx = 0.5 * np.ones_like(zpos)
dy = dx.copy()
dz = f.flatten('F')
ax.bar3d(xpos-0.35, ypos-0.23, zpos, 0.6, 0.6, dz, color=colors)
plt.yticks(np.arange(0, 8), ['AAA', 'AA', 'A', 'BBB', 'BB',
                             'B', 'CCC', 'D'], fontsize=17)
plt.xticks(np.arange(0, 8)[::-1], ['D', 'CCC', 'B', 'BB', 'BBB',
                                   'A', 'AA', 'AAA'], fontsize=17)
ax.set_zlim(0, 0.2)
ax.invert_yaxis()
ax.view_init(38, -129)

add_logo(fig)
