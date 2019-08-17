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

# # S_ProjectionOptionHFP [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_ProjectionOptionHFP&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-proj-hist-dist-fpnew).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import arange, reshape, zeros, where, cumsum, diff, abs, round, mean, log, exp, sqrt, tile, r_, atleast_2d, \
    newaxis, array
from numpy import sum as npsum, max as npmax

from scipy.io import loadmat, savemat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, bar, xlim, ylim, scatter, ylabel, \
    xlabel, title, xticks, yticks

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict
from FPmeancov import FPmeancov
from intersect_matlab import intersect
from PlotTwoDimEllipsoid import PlotTwoDimEllipsoid
from HistogramFP import HistogramFP
from RollPrices2YieldToMat import RollPrices2YieldToMat
from EffectiveScenarios import EffectiveScenarios
from ConditionalFP import ConditionalFP
from Delta2MoneynessImplVol import Delta2MoneynessImplVol
from ColorCodedFP import ColorCodedFP
from HFPquantile import HFPquantile
from InverseCallTransformation import InverseCallTransformation

# parameters
tau = 6  # projection horizon
# -

# ## Upload databases db_ImpliedVol_SPX, db_SwapCurve and db_VIX, and where the common daily observations

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_ImpliedVol_SPX'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_ImpliedVol_SPX'), squeeze_me=True)  # underlying values and implied volatility surface for S&P 500

db_ImpliedVol_SPX = struct_to_dict(db['db_ImpliedVol_SPX'], False)

try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_SwapCurve'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_SwapCurve'),
                 squeeze_me=True)  # rolling values used to computed the short rate

DF_Rolling = struct_to_dict(db['DF_Rolling'], False)

try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_VIX'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_VIX'), squeeze_me=True)  # Vix index values

VIX = struct_to_dict(db['VIX'],False)

# where the common observations between db_ImpliedVol_SPX (thus obtaining a
# reduced db_ImpliedVol_SPX database) and DF_Rolling (thus obtaining a reduced DF_Rolling database)
[_, i_impvol, i_rates] = intersect(db_ImpliedVol_SPX['Dates'], DF_Rolling['Dates'])
db_ImpliedVol_SPX['Dates'] = db_ImpliedVol_SPX['Dates'][i_impvol]
db_ImpliedVol_SPX['Underlying'] = db_ImpliedVol_SPX['Underlying'][i_impvol]
db_ImpliedVol_SPX['Sigma'] = db_ImpliedVol_SPX['Sigma'][:,:, i_impvol]
DF_Rolling['Dates'] = DF_Rolling['Dates'][i_rates]
DF_Rolling['Prices'] = DF_Rolling['Prices'][:, i_rates]

# where the common observations between the reduced db_ImpliedVol_SPX database
# (thus obtaining a new reduced db_ImpliedVol_SPX database) and db_VIX (thus obtaining a reduced db_VIX database)
[dates, i_impvol, i_vix] = intersect(db_ImpliedVol_SPX['Dates'], VIX['Date'])
VIX['Date'] = VIX['Date'][i_vix]
VIX['value'] = VIX['value'][i_vix]
db_ImpliedVol_SPX['Dates'] = db_ImpliedVol_SPX['Dates'][i_impvol]
db_ImpliedVol_SPX['Underlying'] = db_ImpliedVol_SPX['Underlying'][i_impvol]
db_ImpliedVol_SPX['Sigma'] = db_ImpliedVol_SPX['Sigma'][:,:, i_impvol]

# where the observations in the reduced DF_Rolling database which are common
# to the new reduced db_ImpliedVol_SPX database and the reduced db_VIX database
DF_Rolling['Dates'] = DF_Rolling['Dates'][i_impvol]
DF_Rolling['Prices'] = DF_Rolling['Prices'][:, i_impvol]
# -

# ## Extract the risk drivers, i.e. the log value of the underlying, the short shadow rate and the log-implied volatility

# +
# risk driver: the log-value of S&P 500
underlying = db_ImpliedVol_SPX['Underlying']
x_1 = log(underlying)

# risk driver: the short shadow rate
tau_shortrate = 0.3333  # time to maturity of the short rate (4 months)
eta = 0.013  # inverse-call parameter
index_shortrate = where(DF_Rolling['TimeToMat'] == tau_shortrate)
shortrate,_ = RollPrices2YieldToMat(DF_Rolling['TimeToMat'][index_shortrate], DF_Rolling['Prices'][index_shortrate,:])
x_2 = InverseCallTransformation(shortrate, {1:eta}).squeeze()
y = mean(shortrate)

# risk driver: the logarithm of the implied volatility
maturity = db_ImpliedVol_SPX['TimeToMaturity']
delta = db_ImpliedVol_SPX['Delta']  # delta-moneyness
sigma_delta = db_ImpliedVol_SPX['Sigma']
n_ = len(maturity)
k_ = len(delta)
t_x = sigma_delta.shape[2]  # number of risk drivers scenarios

# construct the moneyness grid
max_m = 0.3
min_m = -0.3
n_grid = 6
m_grid = min_m + (max_m - min_m) * arange(n_grid + 1) / n_grid

# m-parametrized log-implied volatility surface
sigma_m = zeros((n_, n_grid + 1, t_x))
for t in range(t_x):
    for n in range(n_):
        sigma_m[n,:,t],*_ = Delta2MoneynessImplVol(sigma_delta[n,:, t], delta, maturity[n], y, m_grid)

x_3 = log(reshape(sigma_m, (n_*(n_grid + 1), t_x),'F'))
# -

# ## Compute the historical daily invariants

# +
epsilon_1 = diff(x_1)
epsilon_2 = diff(x_2)
epsilon_3 = diff(x_3, 1, 1)

t_ = len(epsilon_1)  # number of daily invariants scenarios
# -

# ## Compute the scenarios for the paths of the overlapping invariants for tau=1,...,6

# +
# storage
j_ = t_ - tau + 1  # number of overlapping invariants series
epsilon_1overlap = zeros((j_, tau))
epsilon_2overlap = zeros((j_, tau))
epsilon_3overlap = zeros(((n_grid + 1)*n_, j_, tau))

# overlapping series approach
for j in range(j_):
    # j-th path of the invariants
    epsilon_1overlap[j,:] = cumsum(epsilon_1[j: j + tau])
    epsilon_2overlap[j,:] = cumsum(epsilon_2[j: j + tau])
    epsilon_3overlap[:, j,:] = cumsum(epsilon_3[:, j: j + tau], 1)
# -

# ## Set the Flexible Probabilities via smoothing and scoring on VIX log return
# ## and compute the effective number of scenarios

# +
# VIX value
v_VIX = VIX['value']
# VIX compounded returns
c = diff(log(v_VIX))
# Compute the time series of the risk factor by applying sequentially smoothing and scoringfilters to the time series the VIX index compounded return
# smoothing
z = zeros(t_)
times = range(t_)
tauHL_smoo = 15  # fast half-life time
for t in range(t_):
    p_smoo_t = exp(-log(2) / tauHL_smoo*(tile(t+1, (1, t+1))-times[:t+1]))
    gamma_t = npsum(p_smoo_t)
    z[t] = npsum(p_smoo_t * c[:t+1]) / gamma_t

# scoring
mu_hat = zeros(t_)
mu2_hat = zeros(t_)
sd_hat = zeros(t_)
tauHL_scor = 100  # slow half-life time
for t in range(t_):
    p_scor_t = exp(-log(2) / tauHL_scor*(tile(t+1, (1, t+1))-times[:t+1]))
    gamma_scor_t = npsum(p_scor_t)
    mu_hat[t] = npsum(p_scor_t * z[:t+1]) / gamma_scor_t
    mu2_hat[t] = npsum(p_scor_t * z[:t+1]** 2) / gamma_scor_t
    sd_hat[t] = sqrt(mu2_hat[t]-(mu_hat[t]) ** 2)

z = (z - mu_hat) / sd_hat
z[0] = mu_hat[0]

# conditioner
VIX = namedtuple('VIX', 'Series TargetValue Leeway')
VIX.Series = z.reshape(1,-1)  # time series of the conditioning variable (log return of VIX quotations)
VIX.TargetValue = atleast_2d(z[-1])  # target value for the conditioner
VIX.Leeway = 0.3  # (alpha) probability contained in the range

# prior set of probabilities
tau_HL = 252*4  # (half life) 4 years
prior = exp(-log(2) / tau_HL*abs(arange(VIX.Series.shape[1],0,-1))).reshape(1,-1)
prior = prior / npsum(prior)

# Flexible Probabilities conditioned via entropy pooling
p_all = ConditionalFP(VIX,prior)  # Flexible Probabilities conditioned on the VIX log return, for each day corresponding to the invariants'ime series

p = zeros((1,j_))

for j in range(j_):
    # The flexible probability of the j_th scenario is (proportional to) the average of the probabilities of the tau invariants in the corresponding overlapping series
    p[0,j]=npsum(p_all[0,j:j + tau]) / tau

p = p /npsum(p)

# effective number of scenarios
typ = namedtuple('type','Entropy')
typ.Entropy = 'Exp'
ens = EffectiveScenarios(p, typ)
# -

# ## Compute the scenarios for the paths of the risk drivers by applying the projection formula for tau=1,...,6

x_1hor = x_1[-1] + epsilon_1overlap
x_2hor = x_2[-1] + epsilon_2overlap
x_3hor = tile(x_3[:,[-1],newaxis], [1, j_, tau]) + epsilon_3overlap

# ## Save the data in db_ProjOptionsHFP

# +
# varnames_to_save = [x_1,j_,x_1hor,x_2,x_2hor,x_3,x_3hor,n_,n_grid,tau,eta,sigma_m ,maturity,m_grid,p,ens,sigma_m,dates]
# vars_to_save = {varname: var for varname, var in locals().items() if isinstance(var,(np.ndarray,np.float,np.int)) and varname in varnames_to_save}
# savemat(os.path.join(TEMPORARY_DB, 'db_ProjOptionsHFP'),vars_to_save)
# -

# ## Select the horizon for the plot select the log-underlying and the log- ATM 1yr impl vol compute the HFP mean and covariance

# +
x_1fixhor = x_1hor[:,[-1]]
mateq1 = where(maturity==1)[0]+1
mgrideq0 = where(m_grid==0)[0]+1
x_3fixhor = x_3hor[mateq1*mgrideq0-1,:, [-1]].T

[mu_HFP, sigma2_HFP] = FPmeancov(r_['-1',x_1fixhor, x_3fixhor].T, p)

col = [0.94, 0.3, 0]
colhist = [.9, .9, .9]
# axis settings
x1_l = HFPquantile(x_1fixhor.T, array([[10 ** -6]]), p).squeeze()
x1_u = HFPquantile(x_1fixhor.T, array([[1 - 10 ** -6]]), p).squeeze()
x2_l = HFPquantile(x_3fixhor.T, array([[10 ** -6]]), p).squeeze()
x2_u = HFPquantile(x_3fixhor.T, array([[1 - 10 ** -6]]), p).squeeze()

f = figure()
grey_range = arange(0,0.81,0.01)
CM, C = ColorCodedFP(p, None, None, grey_range, 0, 1, [0.75, 0.25])
# colormap(CM)
option = namedtuple('option', 'n_bins')
option.n_bins = round(6*log(ens))
n1, c1 = HistogramFP(x_1fixhor.T, p, option)
option = namedtuple('option', 'n_bins')
option.n_bins = round(7*log(ens))
n2, c2 = HistogramFP(x_3fixhor.T, p, option)
coeff = 1
plt.subplot2grid((4,4),(1,3),rowspan=3)
plt.barh(c2[:-1], n2[0], height=c2[1]-c2[0], facecolor= colhist, edgecolor= 'k')
plt.axis([0, npmax(n2) + npmax(n2) / 20,x2_l, x2_u])
xticks([])
yticks([])
plt.subplot2grid((4,4),(0,0),colspan=3)
bar(c1[:-1], n1[0], width=c1[1]-c1[0], facecolor= colhist, edgecolor= 'k')
plt.axis([x1_l, x1_u, 0, npmax(n1) + npmax(n1) / 20])
xticks([])
yticks([])
plt.title('Historical Distribution with Flexible Probabilities horizon= {horizon} days'.format(horizon=tau))
plt.subplot2grid((4,4),(1,0),colspan=3, rowspan=3)
X = x_1fixhor
Y = x_3fixhor
scatter(X, Y, 30, c=C, marker='.',cmap=CM)
plt.gca().xaxis.tick_top()
plt.gca().xaxis.set_label_position("top")
xlim([x1_l, x1_u])
ylim([x2_l, x2_u])
xlabel('$X_1$')
ylabel('$x_3$')
plt.gca().yaxis.tick_right()
plt.gca().yaxis.set_label_position("right")
PlotTwoDimEllipsoid(mu_HFP, sigma2_HFP, 1, 0, 0, col, 2);
plt.tight_layout()
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

