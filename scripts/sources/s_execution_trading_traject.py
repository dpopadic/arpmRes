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

# # s_execution_trading_traject [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_execution_trading_traject&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-plopt_-liquidation-trajectories).

# +
import numpy as np
import matplotlib.pyplot as plt

from arpym.portfolio import almgren_chriss
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_execution_trading_traject-parameters)

q_now = 0  # starting volume time
q_end = 1  # ending volume time
h_q_now = 100  # initial holdings
h_q_end = 90  # final holdings
eta = 0.135  # transation price dynamics parameters
sigma = 1.57
lam = np.array([0, 0.3, 1, 5])  # mean-variance trade-off penalties
k_ = 721  # number of grid points [q_now, q_end)

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_execution_trading_traject-implementation-step01): Compute the trading trajectories

# +
l_ = len(lam)
q_grid = np.linspace(q_now, q_end, k_)  # define grid
trajectory = np.zeros((k_, l_))

# Almgren-Chriss trading trajectories
for l in range(l_):
    trajectory[:, l] = almgren_chriss(q_grid, h_q_now, h_q_end, lam[l],
                                      eta, sigma)[0]
# -

# ## Plots

# +
plt.style.use('arpm')

fig = plt.figure()
plt.plot(q_grid, trajectory[:, 0], color='r', label=r'$\lambda$ = 0 (VWAP)')
plt.plot(q_grid, trajectory[:, 1], label='$\lambda$ = 0.3')
plt.plot(q_grid, trajectory[:, 2], color='g', label='$\lambda$ = 1')
plt.plot(q_grid, trajectory[:, 3], color='k', label='$\lambda$ = 5')

plt.axis([q_now, q_end, h_q_end - 1, h_q_now + 1])

plt.xlabel('Volume time')
plt.ylabel('Holdings')
plt.title('Trading trajectories in the Almgren-Chriss model')
plt.legend()
add_logo(fig)
