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

# # S_SpectrumNorm [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_SpectrumNorm&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExSpectrum).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

from numpy import ones, sort, argsort, cov, eye, mean, tile
from numpy.linalg import eig
from numpy.random import randn

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, bar, legend, xlim, ylim, ylabel, \
    xlabel, title, xticks, yticks

plt.style.use('seaborn')

from ARPM_utils import save_plot
from HistogramFP import HistogramFP
from PlotTwoDimEllipsoid import PlotTwoDimEllipsoid
from MarchenkoPastur import MarchenkoPastur
# -

# ## Initialize variables

i_ = 500
t_ = 2000
q = t_ / i_

# ## Generate a time series of (i_ x t_end) observations

X = randn(i_, t_)

# ## Compute the spectrum

Diag_lambda2, e = eig(cov(X))
lambda2_vec, ind = sort(Diag_lambda2)[::-1], argsort(Diag_lambda2)[::-1]  # sorted eigenvalues

# ## Compute the Marchenko-Pastur distribution corresponding to q=t_end/i_

l_ = 100  # coarseness level
x_MP, y_MP, xlim_MP = MarchenkoPastur(q, l_)

# ## Select the entries to plot the ellipsoid
# ## map the sample into the eigenvector space

X_tmp = e[:, ind].T@(X - tile(mean(X, 1,keepdims=True), (1, t_)))
X_ellips = X_tmp[[0,i_-1], :]

# ## Create figures

# +
c0_bl = [0.27, 0.4, 0.9]
c1_or = [1, 0.5, 0.1]
lambda2_min = min(lambda2_vec)
lambda2_max = max(lambda2_vec)
m_lambda2 = lambda2_min - (lambda2_max - lambda2_min) / 10
M_lambda2 = lambda2_max + (lambda2_max - lambda2_min) / 10

# spectrum plot
figure()
# color=w',.Tunits','normalized','outerposition',[0.15, 0.25, 0.4, 0.5])
xlabel(r'Invariants (i)')
ylabel(r'Eigenvalues ($\lambda^2_i$)')
xlim([-50, i_ + 50])
ylim([lambda2_min, 1.2 * lambda2_max])
l1 = plot(range(i_), ones((i_, 1)), color='g', lw=2, label='true spectrum')

l2 = plot(range(i_), lambda2_vec, color=c0_bl, marker='.', label='sample spectrum')
legend()
title('Spectrum');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

# spectrum distribution plot
figure()
# color=w',.Tunits','normalized','outerposition',[0.2, 0.2, 0.4, 0.5])
p = ones((1, len(lambda2_vec))) / len(lambda2_vec)
option = namedtuple('option', 'n_bins')

option.n_bins = 100
density, xbin = HistogramFP(lambda2_vec.reshape(1,-1), p, option)

bar(xbin[:-1], density[0], width=xbin[1]-xbin[0], facecolor=c0_bl, edgecolor=c0_bl)
plot([1, 1], [0, 1], 'g', lw=3)
if q >= 1:
    plot(x_MP, y_MP, color=c1_or, lw=3)
else:
    plot(x_MP[1:l_], y_MP[1:l_], color=c1_or, lw=3)
    plot([x_MP[0], x_MP[0]], [0, y_MP[0]], color=c1_or, lw=6)

xlabel(r'$\lambda^2_i$')
xlim([m_lambda2, M_lambda2])
ylim([0, 1.25 * max(y_MP)])
title('Spectrum distribution')
legend(['sample spectrum', 'true spectrum', 'Marchenko-Pastur']);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

# ellipsoids
figure()
# color=w',.Tunits','normalized','outerposition',[0.25, 0.15, 0.4, 0.5])
S = cov(X_ellips)
S[S < 1e-14] = 0
plot(X_ellips[0], X_ellips[1], '.', markersize=5, color=[0.8, 0.8, 0.8])
# axis square
scale = 2
PlotTwoDimEllipsoid([00], eye(2), scale, 0, 0, 'g', 2,fig=plt.gcf())
PlotTwoDimEllipsoid([00], S, scale, 0, 0, c0_bl, 2,fig=plt.gcf())
xlabel('Variable 1 (rotated)')
ylabel('Variable 2 (rotated)')
legend(['observations', 'true', 'sample'])
title('PCA Ellipsoids')
xlim([-5, 5])
ylim([-5, 5]);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
