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

# # S_ShrinkCovSMT [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_ShrinkCovSMT&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=SMTexe).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import std, mean, log, tile, cov, eye,min as npmin, max as npmax

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, title, subplot, imshow, xticks, yticks

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from SectorSelect import SectorSelect
from ARPM_utils import struct_to_dict
from SMTCovariance import SMTCovariance

# initialize variables
k_ = 20  # number of sparse rotations
# -

# ## Upload database

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_StocksS_P'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_StocksS_P'), squeeze_me=True)

Data = struct_to_dict(db['Data'])
# -

# ## Select the equities belonging to the Materials sector and compute their log returns

sectors = Data.Sectors
index = SectorSelect[sectors, sectors[1]]
i_ = len(index)
v = Data.Prices[index,:]
r = log(v[:, 1:] / v[:, :-1])
t_ = r.shape[1]

# ## Normalize the time series of log-returns

epsi = (r - tile(mean(r, 1,keepdims=True), (1, t_))) / tile(std(r, ddof=1, axis=1), (1, t_))  # normalized time series

# ## Compute the sample covariance matrix from the normalized log-returns

sigma2 = cov(epsi, ddof=1)

# ## Perform shrinkage

sigma2_SMT = SMTCovariance(sigma2, k_)

# ## Create figure

# +
figure()

# gray_mod = gray
max_corr = 0.7
min_corr = npmin(sigma2_SMT)
S = sigma2 - eye(i_)
S[S > max_corr] = max_corr
Corr = sigma2_SMT - eye(i_)
Corr[Corr > max_corr] = max_corr
# plot the sample correlation
subplot(1, 2, 1)
ytlabel = Data.Names[index,:]
xt = i_ + 0.5

imshow(S, [min_corr, max_corr])
# colormap((gray_mod(end: -range(1),:)))
title(['Correlation for sector ', Data.Sectors[1]], )
xticks(range(i_))
yticks(range(i_),ytlabel)
plt.text(range(i_), tile(xt, (i_, 1)), Data.Names[index, :], horizontalalignment='right',rotation=90)
# plot shrunk correlation
subplot(1, 2, 2)
ytlabel = Data.Names[index,:]
xt = i_ + 0.5

imshow(Corr, [min_corr, max_corr])
title('Sparse Matrix Transformation estimate')
xticks(range(i_))
yticks(range(i_), ytlabel)
plt.text(range(i_),tile(xt, (i_, 1)), Data.Names[index, :], horizontalalignment='right',rotation=90)
# number of rotations
D = 'N.of Sparse Rotations = %1.0f'%k_
plt.text(8, -3.5, D, verticalalignment='Bottom',horizontalalignment='Left')
