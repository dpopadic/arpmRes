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

# # S_ShrinkCorrHomClusters [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_ShrinkCorrHomClusters&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=HomClusShrink).

# ## Prepare the environment

# +
import os
import os.path as path
import sys
from collections import namedtuple

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import log, corrcoef, unique, arange, max as npmax, min as npmin, eye, diff, ix_, linspace

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, title, yticks, xticks, imshow, subplot

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import struct_to_dict
from SortBySector import SortBySector
from HomCl import HomCl

# inputs
index = [96, 97, 128, 132, 138]  # entries of interest
# -

# ## Upload database

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_StocksS_P'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_StocksS_P'), squeeze_me=True)

Data = struct_to_dict(db['Data'])
# -

# ## Compute the correlation matrix from the log-returns

# +
prices = Data.Prices

i_ = prices.shape[0]
t_ = prices.shape[1]

epsi = log(prices[:, 1:t_] / prices[:, :t_-1])
c2 = corrcoef(epsi)  # sample correlation matrix
# -

# ## Sort the correlation matrix by sectors

# +
sectors = Data.Sectors
sector_names = unique(sectors)

i_s, l_s = SortBySector(sectors, sector_names)
c2_sec = c2[ix_(i_s.flatten(), i_s.flatten())]  # correlation matrix sorted by sectors
# -

# ## Select the entries of interest and perform homogeneous shrinkage

# +
c2_bar = c2_sec[ix_(index, index)]

options = namedtuple('options', 'method i_c l_c')
options.method = 'exogenous'
options.i_c = range(5)
options.l_c = [0, 2, 5]
c2_hom = HomCl(c2_bar, options)[0]
# -

# ## Create figure

# +
c_gray = [0.8, 0.8, 0.8]

gray_mod = c_gray

tick = l_s[:-1]+diff(l_s) / 2
rho2_f = c2_sec - eye(i_)
c_max = npmax(rho2_f)
c_min = npmin(rho2_f)

f, ax = plt.subplots(1,2)
plt.sca(ax[0])
ytlab = arange(5)
cax = imshow(c2_bar,aspect='equal')
cbar = f.colorbar(cax,ticks=linspace(c_min,c_max,11),format='%.2f',shrink=0.53)
plt.grid(False)
# colormap gray
xticks(arange(5),arange(1,6))
yticks(arange(5),arange(1,6))
title('Starting Correlation')
plt.sca(ax[1])
ax[1].set_adjustable('box-forced')
cax1 = imshow(c2_hom, aspect='equal')
cbar = f.colorbar(cax1,ticks=linspace(c_min,c_max,11),format='%.2f',shrink=0.53)
plt.grid(False)
yticks([])
xticks(arange(5),arange(1,6))
title('Homogenized Correlation');

