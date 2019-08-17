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

# # S_MissingObservations [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_MissingObservations&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerMissingObs).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import arange, ones, percentile, floor, diff, abs, exp, r_, ix_, array, zeros
from numpy import sum as npsum
from numpy.random import randint, choice

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, xlim, ylim, scatter, ylabel, \
    xlabel
import matplotlib.dates as mdates

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict, date_mtop
from intersect_matlab import intersect
from PlotTwoDimEllipsoid import PlotTwoDimEllipsoid
from RollPrices2YieldToMat import RollPrices2YieldToMat
from MaxLikelihoodFPLocDispT import MaxLikelihoodFPLocDispT
from ColorCodedFP import ColorCodedFP
from EMalgorithmFP import EMalgorithmFP
# -

# ## Upload dataset

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_SwapCurve'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_SwapCurve'), squeeze_me=True)

DF_Rolling = struct_to_dict(db['DF_Rolling'])
# -

# ## Compute the swap rates daily changes and select the last 700 available observations

# +
# times to maturity (in years)
tau = [1, 2, 3, 5, 7, 8, 10]

# zero rates from rolling pricing
y,_ = RollPrices2YieldToMat(DF_Rolling.TimeToMat, DF_Rolling.Prices)

# select zero rates
_, _, tauIndices = intersect(tau, DF_Rolling.TimeToMat)
y_tau = y[tauIndices, :]

dates = DF_Rolling.Dates

# daily changes (last 700 obs available)
i_ = len(tau)
t_ = 700

dy = diff(y_tau, 1, 1)
dy = dy[:, - t_:]
dates = dates[- t_:]
# -

# ## Maximum likelihood with Flex. Probs. - complete series

# +
nu = 4
lam = 0.002
flex_prob = exp((-lam * arange(t_, 1 + -1, -1))).reshape(1,-1)
flex_prob = flex_prob / npsum(flex_prob)
tol = 10 ** -5
mu_all, s2_all, err_all = MaxLikelihoodFPLocDispT(dy, flex_prob, nu, tol, 1)

epsi_25 = dy[[1, 3],:]
mu_all_25 = mu_all[[1, 3]]
s2_all_25 = s2_all[np.ix_([1, 3], [1, 3])]
# -

# ## Missing data: randomly drop observations (jointly from the 2 and 5 years series)

# +
# randomly drop 15% of the obs from the whole dataset
ndrop = int(floor(0.15*t_))
Drop_idx = zeros((7,int(ndrop/7)),dtype=int)
for i in range(7):
    Drop_idx[i] = choice(arange(t_), size=int(ndrop/7), replace=False)
epsi = dy.copy()
for i in range(7):
    epsi[i,Drop_idx[i]] = np.NAN

# restore the observations dropped from the 2 and 5 year series and jointly
# drop 30# of the observations from them
epsi[1] = dy[1].copy()
epsi[3] = dy[3].copy()

ndrop_25 = int(floor(0.3*t_))
drop_25 = randint(0,t_-1, size=ndrop_25)
epsi[1, drop_25] = np.NAN
epsi[3, drop_25] = np.NAN

# identify available [a] and not-available (na) data

a = ones((i_, t_))
a[np.arange(0,7).reshape(-1,1),Drop_idx] = 0
a[1] = ones((1, t_))
a[3] = ones((1, t_))
a[1, drop_25] = 0  # a((available obs))=1 a((not-available obs))=0
a[3, drop_25] = 0  # a((available obs))=1 a((not-available obs))=0

na = abs(a - 1)  # na((available obs))=0 na((not-available obs))=1
# -

# ## EM algorithm for Maximum Likelihood with Flexible Probabilities (EMFP estimators)

# +
mu_EMFP, s2_EMFP = EMalgorithmFP(epsi, flex_prob, nu, tol)

# EMFP estimators for 2 and 5 years swap rate daily changes
mu_EMFP_25 = mu_EMFP[[1, 3]]
s2_EMFP_25 = s2_EMFP[np.ix_([1, 3], [1, 3])]
# -

# ## Truncated series (whenever an observation is missing, the simultaneous observations are dropped)

epsi_trunc = epsi[:, npsum(na,axis=0)==0]
flex_prob_trunc = flex_prob[[0],npsum(na,axis=0)==0].reshape(1,-1)
flex_prob_trunc = flex_prob_trunc / npsum(flex_prob_trunc)

# ## Maximum likelihood with Flex. Probs. - truncated series

# +
mu_trunc, s2_trunc, *_ = MaxLikelihoodFPLocDispT(epsi_trunc, flex_prob_trunc, nu, tol, 1)

# MLFP estimators on the truncated series for 2 and 5 years swap rate daily changes
mu_trunc_25 = mu_trunc[[1, 3]]
s2_trunc_25 = s2_trunc[np.ix_([1, 3], [1, 3])]
# -

# ## Figure

# +
# colors
blue = 'b'
orange = [0.94, 0.35, 0]
green = [0, 0.7, 0.25]

dates_dt = array([date_mtop(i) for i in dates])

# scatter colormap and colors
CM, C = ColorCodedFP(flex_prob, None, None, arange(0.25,0.81,0.01), 0, 18, [12, 0])

figure()
myFmt = mdates.DateFormatter('%d-%b-%Y')
# colormap(CM)
# upper plot: scatter plot and ellipsoids
plt.subplot2grid((4,1),(0,0),rowspan=3)
# scatter plot
ss = scatter(epsi_25[0], epsi_25[1], s=20, c=C, marker='o')
xlim([percentile(epsi_25[0], 100*0.05), percentile(epsi_25[0], 100*0.95)])
ylim([percentile(epsi_25[1], 100*0.05),percentile(epsi_25[1], 100*0.95)])
xlabel('2yr rate daily changes')
ylabel('5yr rate daily changes')
# ellipsoids
ell1 = PlotTwoDimEllipsoid(mu_EMFP_25.reshape(-1,1), s2_EMFP_25, r=1, color=orange, linewidth=2.3)
ell = PlotTwoDimEllipsoid(mu_all_25.reshape(-1,1), s2_all_25, r=1, color=blue, linewidth=2.9)
ell2 = PlotTwoDimEllipsoid(mu_trunc_25.reshape(-1,1), s2_trunc_25, r=1, color=green, linewidth=2.7)
# highlight the dropped obs in the scatter plot (white circles)
dr = plot(epsi_25[0, drop_25], epsi_25[1, drop_25],markersize=5,markeredgecolor='k',marker='o',
          markerfacecolor= [0.9, 0.7, 0.7],linestyle='none')
# leg
leg = legend(['Expectation-Maximization w. FP','MLFP - complete series','MLFP - truncated series','Dropped obs'])
# bottom plot: highlight missing observations in the dataset as white spots
ax = plt.subplot(4,1,4)
ax.imshow(np.flipud(abs(na-1)), extent=[dates_dt[0].toordinal(),dates_dt[-1].toordinal(),0, 8], aspect='auto')
plt.yticks([2,4],['2yr','5yr'])
plt.xticks(dates_dt[np.arange(49,t_-2,200,dtype=int)])
ax.xaxis.set_major_formatter(myFmt)
ax.invert_yaxis()
plt.grid(False)
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
