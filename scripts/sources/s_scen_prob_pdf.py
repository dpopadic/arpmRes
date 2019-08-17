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

# # s_scen_prob_pdf [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_scen_prob_pdf&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-biv-fpexample).

# +
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from arpym.estimation import cov_2_corr
from arpym.statistics import meancov_sp, pdf_sp
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_scen_prob_pdf-parameters)

h2 = 0.01  # bandwidth
x = np.array([[0, 0, 1, 1], [0, 1, 0, 1]]).T  # joint scenarios
p = np.array([0.33, 0.10, 0.20, 0.37])  # probabilities

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_scen_prob_pdf-implementation-step01): Compute expectation and covariance

m, s2 = meancov_sp(x, p)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_scen_prob_pdf-implementation-step02): Compute correlation matrix

c2, _ = cov_2_corr(s2)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_scen_prob_pdf-implementation-step03): Compute the scenario-probability pdf

# +
# grid points for pdf evaluation
x1_grid = np.arange(np.min(x[:, 0])-0.5, np.max(x[:, 0])+0.5, 0.025)
x2_grid = np.arange(np.min(x[:, 1])-0.5, np.max(x[:, 1])+0.5, 0.025)
x_grid = np.array([[x1, x2] for x1 in x1_grid for x2 in x2_grid])

# scenario-probability pdf
f = pdf_sp(h2, x_grid, x, p)
# -

# ## Plots

# +
# figure settings
plt.style.use('arpm')
fig = plt.figure()
ax = fig.gca(projection='3d')

# pdf surface
x_1, x_2 = np.meshgrid(x1_grid, x2_grid)
l_ = len(x1_grid)
m_ = len(x2_grid)
ax.plot_surface(x_1, x_2, f.reshape(l_, m_).T, linewidth=0.3, color='w', edgecolors='black')

# ticks and labels
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
ax.set_xticks(np.sort(x[:,0]))
ax.set_yticks(np.sort(x[:,1]))

add_logo(fig)
