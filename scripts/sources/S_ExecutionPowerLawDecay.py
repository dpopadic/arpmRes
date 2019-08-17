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

# # S_ExecutionPowerLawDecay [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_ExecutionPowerLawDecay&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-liquidation_power_law_decay).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import arange

from scipy.special import beta as betafunc, betainc

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, legend, subplots, ylabel, \
    xlabel, title

plt.style.use('seaborn')
np.seterr(divide='ignore')

from ARPM_utils import save_plot

# parameters
h_start = 10000  # initial holdings
delta_q = 0.01  # time interval occurring between two trades
q_grid = arange(0,1+delta_q,delta_q)
omega1 = 0.5
omega2 = 0.8
omega3 = 0.2
# -

# ## Compute the optimal trading rate and the optimal trading trajectory

# +
# corresponding to omega=0.2,0.5,0.8

# optimal trajectories
trajectory1 = h_start*(
1 - betainc((1 + omega1) / 2, (1 + omega1) / 2,q_grid) / betainc((1 + omega1) / 2, (1 + omega1) / 2, 1))
trajectory2 = h_start*(
1 - betainc((1 + omega2) / 2, (1 + omega2) / 2, q_grid) / betainc((1 + omega2) / 2, (1 + omega2) / 2, 1))
trajectory3 = h_start*(
1 - betainc((1 + omega3) / 2, (1 + omega3) / 2, q_grid) / betainc((1 + omega3) / 2, (1 + omega3) / 2, 1))

# optimal trading rates
trading_rate1 = -h_start / (betafunc((1 + omega1) / 2, (1 + omega1) / 2)*(q_grid * (1 - q_grid)) ** (1 - omega1 / 2))
trading_rate2 = -h_start / (betafunc((1 + omega2) / 2, (1 + omega2) / 2)*(q_grid * (1 - q_grid)) ** (1 - omega2 / 2))
trading_rate3 = -h_start / (betafunc((1 + omega3) / 2, (1 + omega3) / 2)*(q_grid * (1 - q_grid)) ** (1 - omega3 / 2))
# -

# ## Plot the optimal trading rate and the optimal trading trajectory for each value of omega.

# +
f,ax = subplots(2,1)

# optimal trading trajectories
plt.sca(ax[0])
title('Transient impact: power law decay kernel')
a1 = plot(q_grid, trajectory1, color='r',label='$\omega$ = 0.5')
a2 = plot(q_grid, trajectory2, color='b',label='$\omega$ = 0.8')
a3 = plot(q_grid, trajectory3, color='k',label='$\omega$ = 0.2')
leg1 = legend()
ylabel( 'Share holdings')

# optimal trading rates
plt.sca(ax[1])
p1 = plot(q_grid, trading_rate1, color='r',label='$\omega$ = 0.5')
p2 = plot(q_grid, trading_rate2, color='b',label='$\omega$ = 0.8')
p3 = plot(q_grid, trading_rate3, color='k',label='$\omega$ = 0.2')
leg2 = legend()
xlabel('Volume time')
ylabel('rading rate')
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
