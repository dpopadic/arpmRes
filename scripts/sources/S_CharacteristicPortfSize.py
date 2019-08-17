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

# # S_CharacteristicPortfSize [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_CharacteristicPortfSize&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-ch-portfolio-size).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import arange, trace, array, zeros, sort, argsort, percentile, linspace, cov, diag, eye, abs, round, mean, log, \
    sqrt, r_
from numpy import min as npmin, max as npmax
np.seterr(divide='ignore',invalid='ignore')

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, bar, subplots, ylabel, \
    xlabel, title, xticks, yticks
import matplotlib.dates as mdates

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, date_mtop
from EwmaFP import EwmaFP
from cov2corr import cov2corr
from PnlStats import PnlStats
from FactorReplication import FactorReplication
from pcacov import pcacov
# -

# ## Upload the database

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_signals_size'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_signals_size'), squeeze_me=True)

v = db['v']
t_start = db['t_start']
t_scor = db['t_scor']
t_smoo = db['t_smoo']
dates = db['dates']
s_siz_rk = db['s_siz_rk']

# settings
n_, _ = v.shape
t_ = v.shape[1]  # final date
nu_x = log(2) / 100  # half-life for HFP covariance matrix of compounded returns
nu_ic = log(2) / 40  # half-life information coefficient
dates = array([date_mtop(int(i)) for i in dates[t_start:t_]])
s = s_siz_rk[:, t_start - t_smoo - t_scor+1:]

tsize = t_-t_start

sigma2_pnl_shr = zeros((n_,n_,tsize))
sigma2_pnl = zeros((n_,n_,tsize))
sigma2_h = zeros((1,tsize))
beta = zeros((n_,tsize))
h_mv = zeros((n_,tsize))
market_pnl = zeros((n_,tsize))
pnl_contr = zeros((n_,tsize))
ic_hat = zeros((1,tsize))
x = zeros((21,tsize))
y = zeros(x.shape)

for t in range(t_start, t_):
    # ## Estimate the P&L covariance matrix

    # compute the HFP covariance matrix of compounded returns with exponentially decaying Flexible Probabilities
    epsi = log(v[:, t - t_start + 1:t] / v[:, t - t_start:t - 1])
    _, sigma2_epsi = EwmaFP(epsi, nu_x)
    sigma2_pnl[:, :, t - t_start] = np.diagflat(v[:, t-1])@sigma2_epsi@np.diagflat(v[:, t-1])  # non-shrinked cov matrix
    # compute the shrinked HFP covariance matrix
    s_epsi, c2_epsi = cov2corr(sigma2_epsi)
    gamma = 0.7
    c2_epsi = (1 - gamma) * c2_epsi + gamma * eye(n_)
    sigma2_epsi_bar = np.diagflat(s_epsi)@c2_epsi@np.diagflat(s_epsi)
    sigma2_pnl_shr[:, :, t - t_start] = np.diagflat(v[:, t-1])@sigma2_epsi_bar@np.diagflat(v[:, t-1])  # shrinked cov matrix

    # ## Compute the characteristic portfolio and the realized portfolio P&L contributions

    sigma_vec_x = sqrt(diag(sigma2_epsi_bar))
    beta[:, t - t_start] = v[:, t-1] * sigma_vec_x * s[:, t - t_start]
    h_mv[:, t - t_start] = FactorReplication(beta[:, t - t_start], sigma2_pnl_shr[:, :, t - t_start])
    market_pnl[:, t - t_start] = v[:, t] - v[:, t-1]
    pnl_contr[:, t - t_start] = h_mv[:, t - t_start] * market_pnl[:, t - t_start]  # pnl
    sigma2_pnl[:, :, t - t_start] = np.diagflat(v[:, t-1])@sigma2_epsi@np.diagflat(v[:, t-1])  # non-shrinked cov matrix
    sigma2_h[0,t - t_start] = h_mv[:, t - t_start].T@sigma2_pnl[:, :, t - t_start]@h_mv[:,t - t_start]

    # ## Compute the realized information coefficient

    _, sigma2_pibeta = EwmaFP(r_[market_pnl[:, :t - t_start+1], beta[:, :t - t_start+1]], nu_ic)
    s2_pb = sigma2_pibeta[:n_, n_ :]
    s2_b = sigma2_pibeta[n_:, n_:]
    ic_hat[0,t - t_start] = trace(s2_pb) / trace(s2_b)
# -

# ## Compute the portfolio P&L and some related statistics

stats, dailypnl, cumpnl, highWaterMark, drawdown = PnlStats(pnl_contr)

# ## Compute the best fit regression line of the realized portfolio return contributions t time t+1 against the exposures at time t

real_rets = market_pnl / v[:, t_start-1:t_-1]
exp_rets = beta / v[:, t_start-1:t_-1]
dollar_weights = h_mv * v[:, t_start-1:t_-1]
for t in range(t_ - t_start):
    e, l = pcacov(cov(exp_rets[:, t], real_rets[:, t]))
    m_exp = mean(exp_rets[:, t])
    m_ret = mean(real_rets[:, t])
    alpha_sym = e[:, 1].T@r_[m_ret, m_exp] / e[0, 1]
    beta_sym = -e[1, 1] / e[0, 1]
    x[:, t] = 2 * arange(-10, 11) / 10
    y[:, t] = beta_sym * x[:, t] + alpha_sym

# ## Draw the plots

# +
grid_dates = [int(i) for i in linspace(0, len(dates)-1, 5)]

# pnl plot
f, ax1 = subplots(1, 1)
ax2 = ax1.twinx()
ax1.set_ylabel('Cum P&L', color='b')
plt.xticks(dates[grid_dates])
myFmt = mdates.DateFormatter('%d-%b-%y')
ax1.xaxis.set_major_formatter(myFmt)
ax2.set_ylabel('P&L', color='r')
ax2.scatter(dates, dailypnl[0], color='r', linewidth=0, marker='.')
ax2.plot(dates, np.mean(dailypnl) + 2 * sqrt(sigma2_h[0]), color='k')
ax2.plot(dates, np.mean(dailypnl) - 2 * sqrt(sigma2_h[0]), color='k')
ax1.plot(dates, cumpnl.flatten(), color='b', lw=1.5)
ax1.axis([min(dates), max(dates), npmin(cumpnl), npmax(cumpnl)])
title('Characteristic portfolio')
# expected returns vs realized returns
max_abs_rets = max(abs(min(percentile(real_rets, 100 * 0.05,axis=0))), abs(max(percentile(real_rets, 100 * 0.95,axis=0))));
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

figure()
plot(exp_rets[:, -1], real_rets[:, -1], 'b.')
plt.axis([npmin(exp_rets[:, -1]), npmax(exp_rets[:, -1]), -max_abs_rets, max_abs_rets])
plot(x[:, -1], y[:, -1], 'r')
xlabel('Expected returns')
ylabel('Realized returns')

# ordered signal barplot
yy = linspace(1, n_, 5)
stock_tick = round(yy)
signal, idx = sort(s[:, -1]), argsort(s[:, -1])
max_abs_signal = max(abs(min(signal)), abs(max(signal)));
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

figure()
bar(range(1,len(signal)+1),signal)
ylabel('Signals')
plt.axis([0, exp_rets.shape[0] + 1, -max_abs_signal, max_abs_signal])
plt.xticks(stock_tick);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

# exposures plot (sorted wrt the signals)
max_abs_dw = max(abs(min(dollar_weights[:, -1])), abs(max(dollar_weights[:, -1])))

figure()
bar(range(1,len(dollar_weights[idx, -1])+1),dollar_weights[idx, -1])
ylabel('Dollar weights')
plt.axis([0, exp_rets.shape[0] + 1, -max_abs_dw, max_abs_dw])
plt.xticks(stock_tick);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

# normalized beta plot (sorted wrt the signals)
xx = min(beta[idx, -1] / v[idx, -1])
yy = max(beta[idx, -1] / v[idx, -1])
max_abs_beta = max(abs(xx), abs(npmax(yy)));
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

figure()
bar(range(1,len(beta[idx, -1])+1), beta[idx, -1] / v[idx, -1])
ylabel('Characteristics')
plt.axis([0, exp_rets.shape[0] + 1, -max_abs_beta, max_abs_beta])
plt.xticks(stock_tick);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

# information coefficient plot
figure()
plot(dates, ic_hat[0], color='b')
plt.axis([min(dates), max(dates), np.nanmin(ic_hat), np.nanmax(ic_hat)])
ylabel('Information coeff.')
plt.xticks(dates[grid_dates])  # ,.TXTickLabel.T,datestr(dates(grid_dates),.Tdd-mmm-yy')))
myFmt = mdates.DateFormatter('%d-%b-%y')
plt.gca().xaxis.set_major_formatter(myFmt);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

