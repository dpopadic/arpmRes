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

# # S_ProjectionSummaryStats [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_ProjectionSummaryStats&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-proj-summary-statistics).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

from numpy import arange, zeros, diff, abs, log, exp, array, atleast_2d, r_
from numpy import sum as npsum

from scipy.io import loadmat

import matplotlib.pyplot as plt

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import struct_to_dict, datenum
from intersect_matlab import intersect
from EffectiveScenarios import EffectiveScenarios
from ConditionalFP import ConditionalFP
from CentralAndStandardizedStatistics import CentralAndStandardizedStatistics
from ProjectMoments import ProjectMoments
# -

# ## Upload databases

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_OptionStrategy'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_OptionStrategy'), squeeze_me=True)

OptionStrategy = struct_to_dict(db['OptionStrategy'])

try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_VIX'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_VIX'), squeeze_me=True)

VIX = struct_to_dict(db['VIX'])
# -

# ## Compute the invariants (daily P&L) and the time series of the conditioning variable (VIX index)

# +
# invariants (daily P&L)
pnl = OptionStrategy.cumPL  # cumulative P&L
x = diff(pnl)  # daily P&L
dates_x = array([datenum(i) for i in OptionStrategy.Dates])
dates_x = dates_x[1:]

# conditioning variable (VIX)
z = VIX.value
dates_z = VIX.Date

# merging datasets
[dates, i_x, i_z] = intersect(dates_x, dates_z)

pnl = pnl[i_x + 1]
x = x[i_x]
z = z[i_z]
t_ = len(x)
# -

# ## Compute the Flexible Probabilities conditioned via Entropy Pooling

# +
# prior
lam = log(2) / 1800  # half life 5y
prior = exp(-lam*abs(arange(t_, 1 + -1, -1))).reshape(1,-1)
prior = prior / npsum(prior)

# conditioner
VIX = namedtuple('VIX', 'Series TargetValue Leeway')
VIX.Series = z.reshape(1,-1)
VIX.TargetValue = atleast_2d(z[-1])
VIX.Leeway = 0.35

# flexible probabilities conditioned via EP
p = ConditionalFP(VIX, prior)

# effective number of scenarios
typ = namedtuple('type','Entropy')
typ.Entropy = 'Exp'
ens = EffectiveScenarios(p, typ)
# -

# ## Initialize variables and compute the single-period standardized statistics

# +
k_ = 6  # focus on first k_ standardized summary statistics
tau = r_[arange(30,210,30),1000]  # projection horizon

gamma_1, _ = CentralAndStandardizedStatistics(k_, x.reshape(1,-1), p)
# -

# ## Compute and print summary statistics at different horizons tau

# +
gamma_tau = zeros((len(tau), k_))

f_1 = namedtuple('f_1','x p')
f_1.x = x.reshape(1,-1)
f_1.p = p
for h in range(len(tau)):
    gamma_tau[h,:] = ProjectMoments(f_1, tau[h], k_)
print(gamma_tau)
