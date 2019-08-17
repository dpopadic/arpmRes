#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This script computes the joint projected distribution of the P&L's of n_ stocks
# over a one day horizon by applying the historical approach with Flexible Probabilities.
# -

# ## For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-pricing-stocks-hfp).

# +
import os
import os.path as path
import sys
from collections import namedtuple

from scipy.io import savemat

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import ones, mean, std, r_, tile, sum as npsum, min as npmin, max as npmax
from scipy.stats import lognorm

from matplotlib.pyplot import figure, plot, bar, legend

from HistogramFP import HistogramFP
from numpy import arange, abs, log, exp, sqrt

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import title, xlabel, scatter, ylabel, xticks, yticks, subplots

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import struct_to_dict, save_plot
from ColorCodedFP import ColorCodedFP
from EffectiveScenarios import EffectiveScenarios
from PlotTwoDimEllipsoid import PlotTwoDimEllipsoid
from FPmeancov import FPmeancov
# -

# ## Upload database db_StocksS_P

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_StocksS_P'),squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_StocksS_P'),squeeze_me=True)

Data = struct_to_dict(db['Data'])

# stock database which contains the stocks historical values

indexes = [3,4] # indexes of the selected stocks
v = Data.Prices[indexes,:] # historical values corresponding to the stocks quotations
# -

# ## Compute the historical scenarios of the compounded returns

x = log(v) # risk drivers
epsilon = x[:,1:]-x[:,:-1] # invariants
n_,j_ = epsilon.shape

# ## Compute the scenarios of the risk drivers at the horizon (Projection Step)

v_tnow = v[:,[-1]] # current prices
X = log(tile(v_tnow, (1,j_))) + epsilon # projected risk drivers

# ## Find scenarios of the stock's P&L at the horizon (Pricing Step)

V = exp(X) # projected values
Pi = V-tile(v_tnow, (1,j_)) # projected P&L's (it can be computed also as: Pi=tile((v_t, (1,j_))*(exp(X_u-log(tile(v_t, (1,j_))))-1) ))

# ## Set the historical Flexible Probabilities as exponential decay with half life 2 years
# ## and compute the effective number of scenarios by using function EffectiveScenarios

# +
tau_HL = 2*252 # 2 years
p = exp((-log(2))/tau_HL*abs(arange(j_,1+-1,-1))).reshape(1,-1)
p = p/npsum(p)

# effective number of scenarios

typ = namedtuple('type','Entropy')
typ.Entropy = 'Exp'
ens = EffectiveScenarios(p, typ)
# -

# ## Save the data in db_PricEquitiesHFP

vars_to_save = {'n_': n_, 'Pi':Pi, 'ens': ens, 'p':p}
savemat(os.path.join(TEMPORARY_DB,'db_PricStocksHFP'),vars_to_save)

# ## Select two stocks in the portfolio, then create a figure which shows the marginal
# ## distributions of the two stocks and the scatter plot of the stocks's P&L's scenarios

# +
[mu_HFP, sigma2_HFP] = FPmeancov(Pi,p)

col =[0.94, 0.3, 0]
colhist=[.9, .9, .9]

f=figure()

grey_range = arange(0,0.81,0.01)
CM,C = ColorCodedFP(p,None,None,grey_range,0,1,[0.7, 0.2])

option = namedtuple('option', 'n_bins')
option.n_bins = int(round(6*log(ens.squeeze())))
n1,c1 = HistogramFP(Pi[[0]], p, option)
n2,c2 = HistogramFP(Pi[[1]], p, option)

axscatter = plt.subplot2grid((3,3),(1,0),colspan=2,rowspan=2)
scatter(Pi[0],Pi[1], 1, c=C, marker='.', cmap=CM)
xlabel('$\pi_4$')
ylabel('$\pi_5$')
PlotTwoDimEllipsoid(mu_HFP,sigma2_HFP,1,0,0,col,2)

ax = plt.subplot2grid((3,3),(0,0),colspan=2)
bar(c2[:-1],n2[0], width=c2[1]-c2[0],facecolor=colhist, edgecolor='k')
yticks([])
xticks([])
title('Historical Distribution with Flexible Probabilities horizon = 1 day')

ax = plt.subplot2grid((3,3),(1,2),rowspan=2)
plt.barh(c1[:-1], n1[0], height=c1[1] - c1[0], facecolor=colhist, edgecolor='k')
yticks([])
xticks([])
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])


