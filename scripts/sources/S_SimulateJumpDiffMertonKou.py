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

# # S_SimulateJumpDiffMertonKou [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_SimulateJumpDiffMertonKou&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=NormalDoubleExpJumps).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import arange

import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots, title, plot

plt.style.use('seaborn')

from ARPM_utils import save_plot
from JumpDiffusionMerton import JumpDiffusionMerton
from JumpDiffusionKou import JumpDiffusionKou

# initial parameters

tau = 1  # horizon
dt = 1 / 252  # time increment
t = arange(0, tau + dt, dt)  # time vector

j_ = 15  # number of simulated processes
# -

# ## Simulate jump diffusion
# ## arithmetic Brownian motion component

# +
mu_m = -1  # drift
sigma_m = 0.5  # diffusion
# Poisson process component
lambda_m = 5  # intensity
mu_p = 1  # drift of log-jump
sigma_p = 0.2  # diffusion of log-jump

x_m = JumpDiffusionMerton(mu_m, sigma_m, lambda_m, mu_p, sigma_p, t, j_)
# -

# ## Simulate double-exponential

# +
mu_k = 0  # deterministic drift
sigma_k = 0.2  # Gaussian component
lambda_k = 4.25  # Poisson process intensity
p = .5  # probability of up-jump
e1 = 0.2  # parameter of up-jump
e2 = 0.3  # parameter of down-jump

x_k = JumpDiffusionKou(mu_k, sigma_k, lambda_k, p, e1, e2, t, j_)
# -

# ## Generate figure

# +
f, ax = subplots(2, 1)
plt.sca(ax[0])
plot(t, x_m.T)
title('Merton jump-diffusion')

plt.sca(ax[1])
plot(t, x_k.T)
title('double exponential')
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

