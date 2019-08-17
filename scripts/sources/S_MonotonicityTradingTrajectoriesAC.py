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

# # S_MonotonicityTradingTrajectoriesAC [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_MonotonicityTradingTrajectoriesAC&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-acnon-monotonicitynulldrift).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import zeros, linspace

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, subplots, ylabel, \
    xlabel, title

plt.style.use('seaborn')

from ARPM_utils import save_plot
from AlmgrenChrissTrajectory import AlmgrenChrissTrajectory

# Settings
q_start = 0
q_end = 1
k_ = 1000
q_grid = linspace(q_start, q_end, k_)  # (volume) time grid

h_start = 100
h_end = 50

# parameters
lam = 1
eta = 0.135
sigma = 1.57

# inizialization
traj = zeros((2, k_))
# -

# ## Compute the Almgren-Chriss trajectories assuming drift theta=0 and then theta=2@lam@sigma**2@h_

theta = 0
traj[0] = AlmgrenChrissTrajectory(q_grid, h_start, h_end, lam, eta, sigma, 0)[0]
traj[1] = AlmgrenChrissTrajectory(q_grid, h_start, h_end, lam, eta, sigma, 2*lam*sigma**2*h_end)[0]

# ## Plot the two trading trajectories obtained in the previous step

# +
f, ax = subplots(2,1)

plt.sca(ax[0])
p1 = plot(q_grid, traj[0], color = [1, 0, 0])
plt.axis([q_start, q_end, min(traj[0]) - 1, max(traj[:, 0]) + 1])
xlabel( 'Volume time')
ylabel( 'Holdings')
title('Almgren-Chriss trading trajectory with null drift')

plt.sca(ax[1])
p1 = plot(q_grid, traj[1], color = [1, 0, 0])
plt.axis([q_start, q_end, min(traj[0]) - 1, max(traj[:, 0]) + 1])
xlabel('Volume time')
ylabel('Holdings')
title('Almgren-Chriss trading trajectory with adjusted drift')
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
