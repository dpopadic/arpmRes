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

# # S_QuantileApproximationEVT [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_QuantileApproximationEVT&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerEVTIII_old).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import arange, ones, percentile, r_

from scipy.stats import t

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, ylabel, \
    xlabel

plt.style.use('seaborn')

from ARPM_utils import save_plot
from FitGenParetoMLFP import FitGenParetoMLFP
# -

# ## Initialize the parameters

# +
mu = 1
sigma = 2
nu = 7
j_ = 10000

p_bar = 0.05  # EVT threshold
p = arange(0.001, p_bar+0.001, 0.001)  # probability levels
# -

# ## Analytical computation of the quantile's left tail

q_an = mu + sigma*t.ppf(p, nu)

# ## Approximation by simulations

# +
epsi = t.rvs(nu, size=(int(j_ / 2), 1))  # simulations
epsi = r_[epsi, - epsi]  # symmetrize simulations
epsi = mu + sigma*epsi

q_sim = percentile(epsi, p*100)
# -

# ## EVT computation of the quantile's left tail

# +
epsi_bar = percentile(epsi, p_bar*100)
epsi_excess = epsi_bar - epsi[epsi < epsi_bar]

csi, sigma = FitGenParetoMLFP(epsi_excess, ones((1, len(epsi_excess))) / len(epsi_excess))
q_EVT = epsi_bar - (sigma / csi)*((p / p_bar) ** (-csi) - 1)
# -

# ## Generate figure showing the comparison between the estimated quantile functions

# +
figure()

plot(p, q_an, lw=2.5, color=[.3, .3, .3])
plot(p, q_sim, lw=1.7,color=[0.2, .5, 1])
plot(p, q_EVT, lw=1.5,color=[.9, .4, 0])
legend(['exact','simulations','EVT'])
plt.grid(True)
xlabel('confidence p')
ylabel('quantile $q_{\epsilon}(p)$');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
