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

# # S_ExecutionGarleanuPedersen [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_ExecutionGarleanuPedersen&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-trajectories_-garleanu-pedersen).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

from matplotlib.ticker import FormatStrFormatter

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import arange, array

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, ylabel, \
    xlabel, title, xticks

plt.style.use('seaborn')

from ARPM_utils import save_plot
from SolveGarleanuPedersen import SolveGarleanuPedersen

# parameters
s = 3  # number of traded assets
k = 2  # number of predicting factors
h_0 = array([10 ** 5, 10 ** 5, 10 ** 5])  # initial holdings
a_end = 5  # trading days
n = 20  # total number of trades
da = a_end / n  # time interval
b = array([[10.32, 122.34], [145.22, 12.7], [9.3, 100.45]])  # component of the alpha term (together with the trading factors f)
sigma = array([[-1.12, 1, 0.98], [- 1, - 1.40, 1.10], [0.98, - 1.10, 1.50]])  # price variance matrix
omega = sigma@sigma.T
delta = array([[0.12], [0.1]])  # factors variance
Phi2 = array([[0.0064, 0.0180], [0.0180, 0.0517]])  # factors mean reversion
phi2 = array([[0.15, 0.12, 0.3], [0.12, 0.34, 0.45], [0.3, 0.4, 0.98]])  # matrix appearing in the linear market impact
y = 7*10 ** (-5)  # interest rate
lam = 0.02  # risk aversion coefficient
# -

# ## Compute the optimal trading trajectory of the Garleanu-Pedersen model by
# ## using function SolveGarleanuPedersen

epsilon = 0.01  # discretization increment
h = SolveGarleanuPedersen(n, s, k, epsilon, y, lam, omega, phi2, Phi2, delta, b, h_0)

# ## Plot the optimal trading trajectories of the three assets.

# +
figure()

plt.axis([0, 5, 10 ** 5 - 2000, 10 ** 5 + 1500])
a_grid = arange(0,a_end+da,da)
xticks(a_grid)
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%g'))
xlabel('Time periods')
ylabel('Share holdings')
title('Optimal trajectories in the Garleanu-Pedersen model')

p1 = plot(a_grid, h[0], color='b', marker = '.',markersize=15)
p2 = plot(a_grid, h[1], color='r', marker = '.',markersize=15)
p3 = plot(a_grid, h[2], color ='k', marker = '.',markersize=15)
legend(['first asset','second asset','third asset']);
plt.show()
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
