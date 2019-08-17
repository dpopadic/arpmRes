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

# # S_ExecutionLogarithmicDecay [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_ExecutionLogarithmicDecay&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-liquidation_logarithmic_decay).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import arange, pi, sqrt

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, subplots, ylabel, \
    xlabel, title

plt.style.use('seaborn')
np.seterr(divide='ignore')

from ARPM_utils import save_plot

# parameters
h_start = 10000  # initial holdings
delta_q = 0.01  # time interval occurring between two trades
q_grid = arange(0,1+delta_q,delta_q)
# -

# ## Compute the optimal trading rate and the optimal trading trajectory

# optimal trading rate
trading_rate = -h_start / (pi*sqrt(q_grid * (1 - q_grid)))
# optimal trajectory
trajectory = 2*h_start * np.arccos(sqrt(q_grid)) / pi

# ## Plot the optimal trading rate and the optimal trading trajectory.

# +
f,ax = subplots(2,1)

plt.sca(ax[0])
title('Transient impact: logarithmic decay kernel')
p1 = plot(q_grid, trajectory, color='b')
ylabel( 'Share holdings')

plt.sca(ax[1])
p2 = plot(q_grid, trading_rate, color='b')
xlabel('Time')
ylabel('Trading rate')
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
