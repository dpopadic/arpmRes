#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This script computes the distribution of the Apple stock's P&L at the horizon t_hor = t_now+1, starting
# from the historical projected distribution of the equity risk driver,
# i.e. the log-value
# -

# ## For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-pric-stock-hfp).

# +
import os
import os.path as path
import sys
from collections import namedtuple

from scipy.io import savemat

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import ones, mean, std, r_
from scipy.stats import lognorm

from matplotlib.pyplot import figure, plot, bar, legend

from HistogramFP import HistogramFP
from numpy import arange, abs, log, exp, sqrt

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import title, xlabel, scatter, ylabel

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot
from ColorCodedFP import ColorCodedFP
# -

# ## Upload the database db_ProjStockHFP (computed in S_ProjectionStockHFP)

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_ProjStockHFP'),squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_ProjStockHFP'),squeeze_me=True)

p = db['p'].reshape(1,-1)
v = db['v']
x_hor = db['x_hor'].reshape(1,-1)
ens = db['ens']
# -

# ## Compute the stock value at the current time, the stocks's scenarios at the
# ## horizon and the scenario's of the stocks's P&L

v_tnow = v[-1] # stock current value
V = exp(x_hor) # stock's scenarios at the horizon
#P&L's scenarios
pi_hor = V-v_tnow # (or, equivalently, pi_hor = v_tnow@(exp((x_hor-x_tnow)-1)), where x_tnow = x(end)

# ## Show the scatter plot of the stock P&L as a function of the stock value
# ## and the distribution of the stock's P&L at the horizon

# +
# scatter plot stock P&L vs. stock value
figure()

GreyRange=arange(0,0.87,0.01)
CM, C = ColorCodedFP(p,None,None,GreyRange,0,1,[0.7, 0.2])
scatter(V,pi_hor,1,c=C,marker='.',cmap=CM)
xlabel('Stock value')
ylabel('Stock P&L')
title('Scatter plot stock P&L vs. stock value');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

# histogram stock P&L
colhist=[.9, .9, .9]
coledges=[.4, .4, .4]
f=figure()

option = namedtuple('option','n_bins')
option.n_bins = int(round(10*log(ens)))
n,c = HistogramFP(pi_hor, p, option)
hf = bar(c[:-1],n[0], width=c[1]-c[0], facecolor=colhist, edgecolor=coledges)
title('Apple P&L distribution with Flexible Probabilities');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

