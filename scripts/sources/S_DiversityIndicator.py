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

# # S_DiversityIndicator [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_DiversityIndicator&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerCorrDistDiv).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import arange, zeros, diff, abs, log, exp, sqrt, array, r_, corrcoef, tile
from numpy import sum as npsum

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, ylim, title

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import struct_to_dict, save_plot
from ConditionalFP import ConditionalFP
# -

# ## upload data

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_StocksS_P'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_StocksS_P'), squeeze_me=True)

Data = struct_to_dict(db['Data'])
# -

# ## compute the returns on the first 200 stocks in the database (conditioning variables)

# +
ret = diff(log(Data.Prices), 1, 1)

ret = ret[:200,:]
date = Data.Dates[1:]
q_ = ret.shape[0]
t_ = ret.shape[1]
# -

# ## Compute the Flexible probabilities conditioned via Entropy Pooling on each factor

# +
alpha = 0.2

# PRIOR
lam = 0.001
prior = exp(-lam*abs(arange(t_, 1 + -1, -1))).reshape(1,-1)
prior = prior / npsum(prior)

p = zeros((q_,t_))
rho2 = zeros((q_,q_))
distance = zeros((q_,q_))
diversity = zeros(q_)

for q in range(q_):
    z = ret[q,:]

    # conditioner
    Conditioner = namedtuple('conditioner', ['Series', 'TargetValue', 'Leeway'])
    Conditioner.Series = z.reshape(1,-1)
    Conditioner.TargetValue = np.atleast_2d(z[-1])
    Conditioner.Leeway = alpha

    p[q,:] = ConditionalFP(Conditioner, prior)
# -

# ## Battacharayya coeff and Hellinger distances

for q1 in range(q_):
    for q2 in range(q_):
        rho2[q1, q2] = npsum(sqrt(p[q1,:]*p[q2,:]))
        distance[q1, q2] = sqrt(abs(1 - rho2[q1, q2]))

# ## Diversity indicator (UPGMA distance)

for q in range(q_):
    diversity[q] = (1 / (q_-1))*(npsum(distance[q,:])-distance[q, q])

# ## Compute the historical correlation matrix

Hcorr = corrcoef(ret)

# ## Generate the figure

fig = figure()
# historical correlation
ax = plt.subplot2grid((3,9),(1,0),rowspan=2,colspan=4)
im = plt.imshow(Hcorr, aspect='equal')
plt.xticks(r_[array([1]), arange(50, 250, 50)])
plt.yticks(r_[array([1]), arange(50, 250, 50)])
yl = ylim()
plt.grid(False)
plt.title('Historical Correlation')
cax = plt.subplot2grid((3,9),(1,4),rowspan=2,colspan=1)
plt.colorbar(im, cax=cax)
# cb = plt.colorbar(ax1, cax = cax)
# diversity
ax = plt.subplot2grid((3,9),(0,5),rowspan=1,colspan=4)
plt.imshow(tile(diversity.reshape(1,-1),(40,1)))
plt.xticks(r_[array([1]), arange(50, 250, 50)])
plt.yticks([])
plt.title('Diversity')
# Hellinger distance
ax = plt.subplot2grid((3,9),(1,5),rowspan=2,colspan=4)
plt.imshow(distance, aspect='equal')
plt.xticks(r_[array([1]), arange(50, 250, 50)])
plt.yticks(r_[array([1]), arange(50, 250, 50)])
plt.title('Hellinger Distance')
plt.grid(False)
plt.tight_layout(w_pad=-0.1);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

