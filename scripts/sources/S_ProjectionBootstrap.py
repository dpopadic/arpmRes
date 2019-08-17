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

# # S_ProjectionBootstrap [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_ProjectionBootstrap&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExChainHybrHistProj).

# ## Prepare the environment

# +
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import arange, ones, zeros, cumsum, tile, newaxis, r_

import matplotlib.pyplot as plt

plt.style.use('seaborn')

from SampleScenProbDistribution import SampleScenProbDistribution

from S_MultivariateQuest import * # performs the quest for invariance step

# Estimation: We use the historical approach, i.e. we rely on the historical distribution of the invariants epsi

# Projection
tau_proj = 21  # t_hor = tnow + 21 days
# -

# ## Path of the invariants: sampled sequences (bootstrap) approach

j_ = 1000  # number of scenarios
Epsi_path = zeros((d_,tau_proj,j_))
for tau in range(tau_proj):
    Epsi_path[:,tau,:]=SampleScenProbDistribution(epsi, p, j_)

# ## Path of the risk drivers

X_path = zeros((d_, tau_proj + 1, j_))  # initialization
X_path[:, [0],:]=tile(x_tnow[...,newaxis,newaxis], (1, 1, j_))  # first node of the path: current value of the risk drivers

# ## Project stocks and options risk drivers according to a multivariate random walk

RandomWalk_idx = r_[arange(Stocks.i_), arange(Stocks.i_ + Bonds.i_ ,i_)]  # position of the random walk entries in the risk drivers and invariants panels
for j in range(j_):
    X_path[RandomWalk_idx, 1:, j]= tile(X_path[RandomWalk_idx, 0, j][...,newaxis], (1, tau_proj)) + cumsum(Epsi_path[RandomWalk_idx,:, j], 1)

# ## Project the shadow rates according to the VAR(1) model fitted in the quest for invariance step

Rates.idx = arange(Stocks.i_, Stocks.i_+ Bonds.i_)
for j in range(j_):
    for t in range(1,tau_proj + 1):
        X_path[Rates.idx, t, j] = Rates.alpha + Rates.beta@X_path[Rates.idx, [t - 1], j] + Epsi_path[Rates.idx, [t - 1], j]

# ## Probabilities associated to the projected paths

p = ones((1, j_)) / j_
