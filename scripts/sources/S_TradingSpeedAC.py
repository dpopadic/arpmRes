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

# # S_TradingSpeedAC [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_TradingSpeedAC&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-acmonotonicityfullexecution).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import arange, array, zeros

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, legend, xlim, subplots, ylabel, \
    xlabel, title

plt.style.use('seaborn')

from ARPM_utils import save_plot
from AlmgrenChrissTrajectory import AlmgrenChrissTrajectory

# Parameters
q_start = 0
q_end = 1
q_grid = arange(0, q_end+1/(60*120),1/(60*120))
h_start = 100
h_end = 0
lam = 0.3
sigma = 1.57
eta = array([0.01, 0.1, 0.2, 0.8])
l_ = len(eta)
k_ = len(q_grid)

# inizialization
trajectories = zeros((l_, k_))
speed = zeros((l_, k_))
# -

# ## Compute the Almgren-Chriss trading trajectory and the respective trading speed for the four different values of eta

for l in range(l_):
    trajectories[l,:],speed[l, :] = AlmgrenChrissTrajectory(q_grid, h_start, h_end,lam, eta[l], sigma)

# ## Plot the trading trajectories and the respective trading speed computed in the previous step

# +
f, ax = subplots(2,1)
plt.sca(ax[0])
p1 = plot(q_grid, trajectories[0], color=[1, 0, 0])
p2 = plot(q_grid, trajectories[1], color = [1, 0.5, 0])
p3 = plot(q_grid, trajectories[2], color = [0, 0.5, 0])
p4 = plot(q_grid, trajectories[3], color = [0, 0, 1])
plt.axis([q_start, q_end, h_end - 1, h_start + 1])
xlabel('Volume time')
ylabel('Holdings')
title('Trading trajectories in the Almgren-Chriss model as eta varies')
legend(['$\eta$ = 0.01','$\eta$ = 0.1','$\eta$ = 0.2', '$\eta$ = 0.8'])

plt.sca(ax[1])
p1 = plot(q_grid, speed[0], color = [1, 0, 0])
p2 = plot(q_grid, speed[1], color = [1, 0.5, 0])
p3 = plot(q_grid, speed[2], color = [0, 0.5, 0])
p4 = plot(q_grid, speed[3], color = [0, 0, 1])
xlim([q_start, q_end])

xlabel('Volume time')
ylabel('Speed')
title('Trading speeds in the Almgren-Chriss model as eta varies')
legend(['$\eta$ = 0.01','$\eta$ = 0.1','$\eta$ = 0.2', '$\eta$ = 0.8'])
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
