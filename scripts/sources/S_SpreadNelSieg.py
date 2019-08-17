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

# # S_SpreadNelSieg [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_SpreadNelSieg&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerYieldSpread).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

from numpy import array, zeros

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import legend, subplots, ylabel, \
    xlabel

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict
from intersect_matlab import intersect
from RollPrices2YieldToMat import RollPrices2YieldToMat
from BootstrapNelSieg import BootstrapNelSieg

# parameters
par_start = namedtuple('par','theta1 theta2 theta3 theta4_squared')
par_start.theta1 = 0.05  # starting values
par_start.theta2 = 0.05
par_start.theta3 = 0.05
par_start.theta4_squared = 0.05
tau = array([0.0833, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30])  # select time to maturities
# -

# ## Upload the rolling values from db_SwapCurve and compute the corresponding yields to maturity using function RollPrices2YieldToMat

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_SwapCurve'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_SwapCurve'), squeeze_me=True)

DF_Rolling = struct_to_dict(db['DF_Rolling'])

# reference yields from rolling prices
y_ref,_ = RollPrices2YieldToMat(DF_Rolling.TimeToMat, DF_Rolling.Prices)
# -

# ## Upload JPM bond prices from db_CorporateBonds and restrict the yields to available dates

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_CorporateBonds'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_CorporateBonds'), squeeze_me=True)

JPM = struct_to_dict(db['JPM'])

t_ = len(JPM.Date)

# dates extraction
_, _, dateIndices_JPM = intersect(JPM.Date, DF_Rolling.Dates)
y_ref = y_ref[:, dateIndices_JPM]

# Bond schedule
b_sched_JPM = zeros((JPM.Coupons.shape[0],2))
b_sched_JPM[:, 0] = JPM.Coupons/100
b_sched_JPM[:, 1] = JPM.Expiry_Date

# prices
v_bond_JPM = JPM.Dirty_Prices/100
# -

# ## Use function BootstrapNelSieg, which calibrates Nelson-Siegel model on the market prices of JPMorgan coupon-bearing bonds, returns JPMorgan yield curve and, given the reference curve, computes the spread curve

# fitting
_, _, _, _, _, y_JPM, _, y_ref_graph, _, s_JPM, _ = BootstrapNelSieg(JPM.Date, v_bond_JPM, b_sched_JPM, tau, par_start,
                                                                     DF_Rolling.TimeToMat, y_ref)

# ## Plot the reference yield curve and the yield and the spread curve of JPMorgan coupon-bearing bonds

# JPM yield plot
f, ax = subplots(2, 1)
plt.sca(ax[0])
plt.plot(tau, y_JPM[:, t_-1], 'b')
plt.plot(tau, y_ref_graph[:, t_-1], 'r')
xlabel('Time to Maturity')
ylabel('Rate')
legend(['JPM', 'Zero swap'])
plt.grid(True)
plt.xlim([0,30])
plt.ylim([0,0.06])
# JPM spread plot
plt.sca(ax[1])
ax[1].plot(tau, s_JPM[:, t_-1], 'b')
xlabel('Time to Maturity')
ylabel('Spread')
legend(['JPM'])
plt.xlim([0,30])
plt.ylim([0,0.03])
plt.grid(True)
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
