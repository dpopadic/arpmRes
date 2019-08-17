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

# # S_SimulateNIGVG [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_SimulateNIGVG&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=SimulatePureJumpsProcess).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import arange

import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots, title

plt.style.use('seaborn')

from ARPM_utils import save_plot
from Schout2ConTank import Schout2ConTank
from NIG import NIG
from VG import VG

# initial parameters

tau = 1  # horizon
dt = 1 / 252  # time increment
t = arange(0, tau + dt, dt)  # time vector

j_ = 15  # number of simulated processes
# -

# ## Simulate Normal-Inverse-Gaussian process
# ## parameters in Schoutens notation

# +
al = 2.1
be = 0
de = 1
# convert parameters to Cont-Tankov notation
th, k, s = Schout2ConTank(al, be, de)

x_nig = NIG(th, k, s, t, j_)
# -

# ## Simulate Variance-Gamma process

# +
mu = 0.1  # deterministic drift in subordinated Brownian motion
kappa = 1
sigma = 0.2  # s.dev in subordinated Brownian motion

x_vg,_ = VG(mu, sigma, kappa, t, j_)
# -

# ## Generate figure

# +
t = t.reshape(1,-1)
f, ax = subplots(2, 1)
ax[0].plot(t.T, x_nig.T)
title('normal-inverse-Gaussian')

ax[1].plot(t.T, x_vg.T)
title('variance gamma')
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
