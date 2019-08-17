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

# # S_AgnosticCorrPrior [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_AgnosticCorrPrior&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=UninfPrior).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import ones, zeros, eye, round, log, tile
from numpy import min as npmin
from numpy.linalg import eig
from numpy.random import rand

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, bar, title

plt.style.use('seaborn')

from ARPM_utils import save_plot
from HistogramFP import HistogramFP

# Initialize variables
i_ = 3  # dimension of the correlation matirix
k_ = int(i_ * (i_ - 1) / 2)  # number of upper non-diagonal entries
j_ = 10000  # number of simulations
# -

# ## Compute correlations in scenarios

# +
C2 = tile(eye(i_)[..., np.newaxis], (1, 1, j_))
lam = zeros((i_, j_))
Theta = zeros((k_, j_))

j = 1
while j < j_:
    Theta_tilde = 2 * rand(k_, 1) - 1  # generate the uninformative correlations
    k = 0
    for i in range(i_):  # build the candidate matrix
        for m in range(i + 1, i_):
            C2[i, m, j] = Theta_tilde[k]
            C2[m, i, j] = C2[i, m, j]
            k = k + 1

    lam[:, j], _ = eig(C2[:, :, j])  # compute eigenvalues to check positivity

    if npmin(lam[:, j]) > 0:  # check positivity
        Theta[:, [j]] = Theta_tilde  # store the correlations
        j = j + 1
# -

# ## Create figures

# +
# titles
names = {}
k = 0
for i in range(1, i_ + 1):
    for m in range(i + 1, i_ + 1):
        names[k] = r'$\Theta_{%d,%d}$' % (i, m)
        k = k + 1

# univariate marginals
option = namedtuple('option', 'n_bins')
option.n_bins = round(5 * log(j_))
for k in range(k_):
    figure()
    p = ones((1, len(Theta[k, :]))) / len(Theta[k, :])
    n, x = HistogramFP(Theta[[k], :], p, option)
    b = bar(x[:-1], n.flatten(), width=0.95 * (x[1] - x[0]), facecolor=[.7, .7, .7], edgecolor=[1, 1, 1])
    title('histogram of {name}'.format(name=names[k]));
    # save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
