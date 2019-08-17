#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This script projects the swap curve assuming that 6 key shadow rates
# follow a VAR[1-1]/MVOU process and modeling the invariants non
# parametrically through the Historical with Flexible Probabilities
# distribution of the first three principal components.
# Projection is performed via the scenario-based approach,
# resampling from the HFP distribution of the invariants.
# -

# ## For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-icvar-1-pc).

# +
import os
import os.path as path
import sys
from collections import namedtuple

from matplotlib.ticker import FuncFormatter
from numpy.linalg import eig
from numpy.ma import where
from scipy.linalg import expm

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import mean, r_, tile, sum as npsum, min as npmin, max as npmax, diff, percentile, newaxis

from matplotlib.pyplot import figure, plot, axis, grid

from HistogramFP import HistogramFP
from numpy import arange, log, exp, array, zeros, ones

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import title, xlabel, ylabel

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import struct_to_dict, save_plot
from InverseCallTransformation import InverseCallTransformation
from intersect_matlab import intersect
from RollPrices2YieldToMat import RollPrices2YieldToMat
from FitVAR1 import FitVAR1
from VAR1toMVOU import VAR1toMVOU
from MinRelEntFP import MinRelEntFP
from PerpetualAmericanCall import PerpetualAmericanCall
from SampleScenProbDistribution import SampleScenProbDistribution

# -

# ## load database db_SwapCurve and compute the realized time series of weekly rates for the key points of the curve with tau= [1 2 3 5 7 10]

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_SwapCurve'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_SwapCurve'), squeeze_me=True)

DF_Rolling = struct_to_dict(db['DF_Rolling'])

tau = array([1, 2, 3, 5, 7, 10])  # time to maturity (years)
_, _, tauIndices = intersect(tau, DF_Rolling.TimeToMat)
timeStep = 5  # weekly observations
y, _ = RollPrices2YieldToMat(tau, DF_Rolling.Prices[tauIndices, ::timeStep])
dates = DF_Rolling.Dates[1::timeStep]
# -

# ## Compute the corresponding shadow rates, using function InverseCallTransformation with eta=0.013

eta = 0.013
x = InverseCallTransformation(y, {1: eta})

# ## Extract the time series of the realized invariants: fit the VAR[1-1]/MVOU model using functions FitVAR1 and VAR1toMVOU and compute the residuals

# +
print('Quest for invariance: VAR(2)/MVOU fit')
# dx=diff(x,1,2) #increments
# t_=dx.shape[1]
t_ = diff(x, 1, 1).shape[1]

# [mu, theta, sigma2] = FitVAR1MVOU(dx, x(:,1:-1), 1)
[alpha, b, sig2_U] = FitVAR1(x)
# [alpha, b, sig2_U] = FitVAR1(dx, x(:,1:-1))
mu, theta, sigma2, _ = VAR1toMVOU(alpha, b, sig2_U, 1)
epsi = x[:, 1:] - (expm(-theta) @ x[:, :-1]) - tile(mu[..., newaxis], (1, t_))
# -

# ## Dimension reduction: consider the first l_ principal components and compute their HFP distribution

# +
print('Dimension reduction/Estimation: HFP distribution of the first l_ principal components')
# Eigenvalues and eigenvectors
lamda2, e = eig(sigma2)
l2 = lamda2

# Historical scenarios for Z
k_ = len(tau)
l_ = 1  # number of principal components to take into account (l_=1 is set for speed set l_=3 for accurate results)
z = e.T @ epsi
z = z[:l_, :]  # historical scenarios

# determine the Flexible Probabilities via Entropy Pooling starting from an exponential decay prior

# Prior: exponential decay
half_life = 52 * 2  # 2years
prior_decay = log(2) / half_life
p_ = exp(-prior_decay * arange(t_, 1 + -1, -1)).reshape(1, -1)
p_ = p_ / npsum(p_)

# Entropy pooling
# stretched variances
v = l2 * (npsum(l2) / npsum(l2[:l_]))

p = zeros((l_, p_.shape[1]))
for k in range(l_):
    # constraints
    Aeq = r_[z[[k], :] ** 2, ones((1, t_))]
    beq = array([[v[k]], [1]])
    p[k, :] = MinRelEntFP(p_, None, None, Aeq, beq)[0]  # Flexible Probabilities
# -

# ## Project paths for the risk drivers:
# ## resample scenarios for the invariants from the principal components' HFP distribution and feed them into the incremental step projection routine

# +
print('Projection')
u_end = 2 * 52  # max horizon = 2y
u = range(u_end)  # projection to weekly horizons up to u_

j_ = 5000  # number of resampled scenarios

X_u = zeros((x.shape[0], j_, len(u) + 1))
X_u[:, :, 0] = tile(x[:, [-1]], (1, j_))

zsim = zeros((l_, j_, len(u) + 1))
Epsi = zeros((k_, j_, len(u) + 1))

for hor in u:
    # Generate MC simulations for the principal components
    for k in range(l_):
        zsim[k, :, hor + 1] = SampleScenProbDistribution(z[[k], :], p[[k], :], j_)

    # Obtain scenarios for the invariants from the PC scenarios
    Epsi[:, :, hor + 1] = e[:, :l_] @ zsim[:, :, hor + 1]
    # Obtain paths for the risk drivers (shadow rates) from the invariants' scenarios
    X_u[:, :, hor + 1] = expm(-theta) @ X_u[:, :, hor] + tile(mu[..., newaxis], (1, j_)) + Epsi[:, :, hor + 1]
# -

# ## Map the projected risk drivers into the projected term structure of swap rates

Y_u = zeros(X_u.shape)
Y_u[:, :, 0] = tile(y[:, [-1]], (1, j_))
for hor in u:
    for k in range(k_):
        Y_u[k, :, hor + 1] = PerpetualAmericanCall(X_u[k, :, hor + 1].T, {'eta': eta})

# ## Plot the projected distribution of the 5 years yield along with a few simulated paths

# +
pick = where(tau == 5)[0]  # select the 5-year yield
ps = [1, 99]  # quantile levels for the plot

p_flat = ones((1, Y_u[pick, :, 0].shape[1])) / Y_u[pick, :, 0].shape[1]
Y = namedtuple('Y', 'x pdf q m')
Y.x = zeros((1, 81))
Y.pdf = zeros((1, 80))
Y.q = zeros((1, 2))
Y.m = []
option = namedtuple('option', 'n_bins')
for i in range(u_end + 1):
    option.n_bins = 80
    pdf, xx = HistogramFP(Y_u[pick, :, i], p_flat, option)
    q = percentile(Y_u[pick, :, i], ps)  # quantiles
    m = mean(Y_u[pick, :, i])  # mean
    Y.x = r_[Y.x, xx.reshape(1, -1)]
    Y.pdf = r_[Y.pdf, pdf]
    Y.q = r_[Y.q, q.reshape(1, -1)]
    Y.m = r_[Y.m, m]

Y.pdf = Y.pdf[1:]
Y.x = Y.x[1:]
Y.q = Y.q[1:]

# figure
y0 = y[pick, -1]
figure()
plot(arange(0, u_end + 1), Y.q, color='r')
plot(arange(0, u_end + 1), Y.m, color='g')
plot(arange(0, u_end + 1), Y_u[pick, :10, :].squeeze().T, c=[.7, .7, .7], lw=1)
xx = r_[u_end, u_end + Y.pdf[-2] * 0.5, u_end]
yy = r_[npmin(Y.x[-2]), Y.x[-2, :-1], npmax(Y.x[-2])]
plot(xx, yy, 'k', lw=1)
plt.gca().fill_between(xx, yy, color=[.7, .7, .7])
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda z, _: '{:.0%}'.format(z)))
axis([0, npmax(xx) * 1.1, 0, 0.05])
grid(True)
plt.xticks(arange(0, u_end + 1, 52), [0, 1, 2])
title('Projection of the 5yr par swap rates')
xlabel('projection horizon (years)')
ylabel('yields');
plt.show()
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

