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

# # S_OutliersDetection [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_OutliersDetection&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerOutlierDetection).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

from numpy import arange, where, diff, cov, diag, round, mean, log, exp, sqrt, r_
from numpy import sum as npsum, min as npmin, max as npmax
from numpy.random import randint

from scipy.stats import chi2
from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, bar, xlim, ylim, scatter, ylabel, \
    xlabel

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict
from PlotTwoDimEllipsoid import PlotTwoDimEllipsoid
from HistogramFP import HistogramFP
from RollPrices2YieldToMat import RollPrices2YieldToMat
from ColorCodedFP import ColorCodedFP
from DetectOutliersFP import DetectOutliersFP
from SpinOutlier import SpinOutlier
from HighBreakdownFP import HighBreakdownFP
# -

# ## Upload dataset

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_SwapCurve'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_SwapCurve'), squeeze_me=True)

DF_Rolling = struct_to_dict(db['DF_Rolling'])
# -

# ## Compute the swap rates daily changes and select the last 500 available observations

# +
# swap rates
dates = DF_Rolling.Dates
ZeroRates,_ = RollPrices2YieldToMat(DF_Rolling.TimeToMat, DF_Rolling.Prices)

y1 = ZeroRates[DF_Rolling.TimeToMat == 1,:]  # 1 year yield
y3 = ZeroRates[DF_Rolling.TimeToMat == 3,:]  # 3 years yield

# daily changes
dy1 = diff(y1)
dy1 = dy1[:, :400]

dy3 = diff(y3)
dy3 = dy3[:, :400]

# invariants
epsi = r_[dy1, dy3]
i_, t_ = epsi.shape
# -

# ## Generate outliers using the function SpinOutlier and add one of them to the dataset.

# +
# Apply DetectOutliersFP and check that the artificial outlier is detected by the test

print('Add and detect outlier')

outliers = SpinOutlier(mean(epsi, 1, keepdims=True), cov(epsi), 2.5, 5)  # generate 5 outliers along a circle centered in the sample mean
outlier = outliers[:, [randint(0 , 5)]]  # choose randomly one of the outliers

epsi_out = r_['-1',epsi, outlier]
# -

# ## Set the Flexible probabilities (exponential decay)

lam = 0.001
p = exp(-lam * arange(t_ + 1, 1 + -1, -1)).reshape(1,-1)
p = p /npsum(p)

# ## Estimate the expectation and covariance based on the robust HBFP estimators

# +
c = 0.75
mu_HBFP, sigma2_HBFP,*_ = HighBreakdownFP(epsi_out, p.copy(), 1, c)

# Rescale the HBFP dispersion parameter to obtain an estimate of the covariance (rescaling constant set based on multivariate normality)
sigma2 = sigma2_HBFP/ chi2.ppf(c, i_)
# -

# ## Univariate analysis: compute the marginal distributions and the z-scores

# +
option = namedtuple('option', 'n_bins')
option.n_bins = round(10*log(t_))
p1, x1 = HistogramFP(epsi_out[[0]], p /npsum(p), option)
p2, x2 = HistogramFP(epsi_out[[1]], p /npsum(p), option)

# z-scores
sdev = sqrt(diag(sigma2))
z_scores = (outlier - mu_HBFP.reshape(-1,1))/sdev.reshape(-1,1)
# -

# ## Multivariate analysis: outlier detection with FP (Mahalanobis distance test)

# +
q = 0.975
[position_outliers, detected_outliers, MahalDist] = DetectOutliersFP(epsi_out, p.copy(), q)

# Find in the output the outlier we added
i = where(position_outliers == t_)[0]
# Mahalanobis distance of the outlier
Mah = MahalDist[i]
# -

# ## Figure

orange = [0.94, 0.35, 0]
grey = [.8, .8, .8]
green = [0, 0.8, 0]
# scatter colormap and colors
[CM, C] = ColorCodedFP(p, None, None, arange(0,0.81,0.01), 0, 1, [0.6, 0.2])
figure()
# colormap(CM)
# marginal of epsi_2 (change in 3yr yield)
ax = plt.subplot2grid((4,5),(1,0),rowspan=3,colspan=1)
plt.barh(x2[:-1], p2[0], height=x2[1]-x2[0], facecolor= grey, edgecolor= 'k')  # histogram
plot([0,0], [mu_HBFP[1] - sdev[1], mu_HBFP[1] + sdev[1]], color=orange,lw=5)  # +/- standard deviation bar
plot(0, epsi_out[1,-1], color='b',marker='o',markerfacecolor='b', markersize = 5)  # outlier
plt.ylim([npmin(epsi_out[1]), npmax(epsi_out[1])])
plt.xticks([])
plt.yticks([])
ax.invert_xaxis()
# marginal of epsi_1 (change in 1yr yield)
ax = plt.subplot2grid((4,5),(0,1),rowspan=1,colspan=4)
bar(x1[:-1], p1[0], width=x1[1]-x1[0], facecolor= grey, edgecolor= 'k')
plt.xticks([])
# # histogram
plot([mu_HBFP[0] - sdev[0], mu_HBFP[0] + sdev[0]], [0,0], color=orange, lw=5)
# # +/- standard deviation bar
plot(epsi_out[0,-1], 0, color='b',marker='o',markerfacecolor='b', markersize = 5)  # outlier
xlim([min(min(epsi[0]), epsi_out[0,-1]), max(max(epsi[0]), epsi_out[0,-1])])
ax = plt.subplot2grid((4,5),(1,1),rowspan=3,colspan=4)
# scatter-plot with HBFP-ellipsoid superimposed, artificial outlier, detected outliers
scatter(epsi_out[0], epsi_out[1], 20, c=C, marker='.',cmap=CM)
# # scatter-plot
ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
PlotTwoDimEllipsoid(mu_HBFP.reshape(-1,1), sigma2, 1, 0, 0, orange, 2.5)  # HBFP ellipsoid
PlotTwoDimEllipsoid(mu_HBFP.reshape(-1,1), sigma2, sqrt(chi2.ppf(q, i_)), 0, 0, green, 0.5)  # threshold
plot(epsi_out[0,-1], epsi_out[1,-1], color='b',marker='o',markersize=4,markerfacecolor='b') #artificial outlier
scatter(epsi_out[0, position_outliers], epsi_out[1, position_outliers], 30, marker='o', edgecolor=green, facecolor='none')
# # detected outliers
plt.xlim([min(min(epsi[0]), epsi_out[0,-1]), max(max(epsi[0]), epsi_out[0,-1])])
plt.ylim([min(min(epsi[1]), epsi_out[1,-1]), max(max(epsi[1]), epsi_out[1,-1])])
x_lim = plt.xlim()
y_lim = plt.ylim()
xl = xlabel('$\epsilon_1$')
yl = ylabel('$\epsilon_2$')
# # standard deviations lines
plot([mu_HBFP[0] - sdev[0], mu_HBFP[0] - sdev[0]], [mu_HBFP[1] - sdev[1], max(epsi[1])],linestyle='--',color=orange,lw=1.3)
plot([mu_HBFP[0] + sdev[0], mu_HBFP[0] + sdev[0]], [mu_HBFP[1] - sdev[1], y_lim[1]],linestyle='--',color=orange,lw=1.3)
plot([x_lim[0], mu_HBFP[0] + sdev[0]], [mu_HBFP[1] - sdev[1], mu_HBFP[1] - sdev[1]],linestyle='--',color=orange,lw=1.3)
plot([x_lim[0], mu_HBFP[0] + sdev[0]], [mu_HBFP[1] + sdev[1], mu_HBFP[1] + sdev[1]],linestyle='--',color=orange,lw=1.3)
# # text boxes
Dist = 'Mahalanobis distance =  % 3.2f' % Mah
plt.text(0.9*x_lim[1], 0.9*y_lim[0], Dist,
             color='b',horizontalalignment='right',verticalalignment='bottom')
sdev1 = '%+.2f s.dev.'% z_scores[0]
plt.text(x_lim[1], y_lim[1], sdev1, color='k',horizontalalignment='right',verticalalignment='bottom')
sdev2 = '%+.2f s.dev.'% z_scores[1]
plt.text(0.99*x_lim[0], y_lim[1], sdev2, color='k',horizontalalignment='left',verticalalignment='top',rotation=90)
plt.text(mu_HBFP[0], 0.9*y_lim[1],'+ / - s.dev.',color=orange,horizontalalignment='center',verticalalignment='bottom')
plt.text(.95*x_lim[0], mu_HBFP[1],' + / - s.dev.',color=orange,horizontalalignment='center',verticalalignment='center', rotation=90)
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
