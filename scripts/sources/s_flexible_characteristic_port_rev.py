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

# # s_flexible_characteristic_port_rev [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_flexible_characteristic_port_rev&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-mom-signal-copy-2).

# +
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from arpym.statistics import ewm_meancov, meancov_sp
from arpym.tools import pca_cov, add_logo
from arpym.portfolio import char_portfolio
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_flexible_characteristic_port_rev-parameters)

tau_hl = 100  # hl used to estimate (cond.) covariance of instruments
tau_hl_lam = 40  # hl used to compute the realized factor premium
n_ = 30  # number of instruments (set low for speed)
v_0 = 0  # budget constraint
sig2pl_max = 0.066  # upper bound for char. port. variance

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_flexible_characteristic_port_rev-implementation-step00): load data

# +
path = '../../../databases/temporary-databases/'
s_mom_rk = pd.read_csv(path + 's_mom_rk.csv', index_col=0, parse_dates=True)
dates = pd.to_datetime(s_mom_rk.index).date
s_mom_rk = np.array(s_mom_rk)
path1 = '../../../databases/global-databases/strategies/db_strategies/'
v = pd.read_csv(path1 + 'last_price.csv', index_col=0, parse_dates=True)
v = np.array(v)
v = v[:, :n_]
t_ = s_mom_rk.shape[0]

db = pd.read_csv(path + 'db_char_port.csv', parse_dates=True)
h_mv = np.array(db['h_char'])
n_mkt = np.array(db['n_'][0])
h_shape = h_mv.shape[0] / n_mkt
h_shape = h_shape.astype(int)
n_mkt = n_mkt.astype(int)
h_mv = h_mv.reshape(h_shape, n_mkt)
h_mv = h_mv[:, :n_]
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_flexible_characteristic_port_rev-implementation-step01): reversal momentum signal

s = -s_mom_rk[:, :n_]

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_flexible_characteristic_port_rev-implementation-step02): (conditioned) covariance matrix of risk drivers

# +
sig2_pl = np.zeros((t_, n_, n_))
sig_vol_hat_x = np.zeros((t_, n_))

for t in np.arange(0, t_):
    w_shift = 252*2  # rolling window
    epsi = np.diff(np.log(v[t:t + w_shift, :]), axis=0)
    _, sig2_hat_x = ewm_meancov(epsi, tau_hl)
    sig_vol_hat_x[t, :] = np.sqrt(np.diag(sig2_hat_x))

    # (conditioned) covariance matrix of P&L via Taylor approximation
    delta = np.diag(v[t + w_shift, :])
    sig2_pl[t, :, :] = delta @ sig2_hat_x @ delta.T
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_flexible_characteristic_port_rev-implementation-step03): signal characteristics

# +
beta = np.zeros((t_, n_))

for t in np.arange(0, t_):
    beta[t, :] = v[t + w_shift, :] * sig_vol_hat_x[t, :] * \
                          s[t, :]
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_flexible_characteristic_port_rev-implementation-step04): characteristic portfolio

# +
h_char = np.zeros((t_, n_))

for t in np.arange(0, t_):
    h_0 = h_mv[t, :]
    h_tilde = 1 / (n_ * v[t + w_shift - 1, :])
    h_char[t, :] = char_portfolio(beta[[t], :].T, sig2_pl[t, :, :],
                                  h_0, h_tilde, sig2pl_max,
                                  v[t + w_shift - 1, :], v_0).T
# -

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_flexible_characteristic_port_rev-implementation-step05): realized characteristics portfolio P&L and its std. dev.

# +
z_char = np.zeros(t_)
sig2_pl_h_real = np.zeros(t_)
pl_real = v[1:, :] - v[:-1, :]

for t in np.arange(0, t_):
    z_char[t] = h_char[t, :].T @ \
        pl_real[t - 1, :]
    sig2_pl_h_real[t] = h_char[t, :].T @ \
        sig2_pl[t, :, :] @ \
        h_char[t, :]
# -

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_flexible_characteristic_port_rev-implementation-step06): realized factor premium

lambda_hat = np.zeros(t_)
for t in range(0, t_):
    lambda_hat[t], _ = ewm_meancov(z_char[:t + 1], tau_hl_lam)

# ## [Step 7](https://www.arpm.co/lab/redirect.php?permalink=s_flexible_characteristic_port_rev-implementation-step07): return vs expected returns, symm. regression line

# +
exp_ret = lambda_hat[-1] * beta[-1, :] / v[-2, :]
real_ret = pl_real[-1, :] / v[-2, :]

# symmetric regression
exp_real = np.concatenate((exp_ret.reshape(-1, 1), real_ret.reshape(-1, 1)),
                          axis=1)
mu_exp_real, sig2_exp_real = meancov_sp(exp_real)
e, _ = pca_cov(sig2_exp_real)
mu_real = mu_exp_real[1]
mu_exp = mu_exp_real[0]
beta_sym = -e[1, 1] / e[0, 1]
alpha_sym = mu_exp - beta_sym*mu_real
x = 2 * np.arange(-10, 11) / 10
y = beta_sym * x + alpha_sym
# -

# ## Save characteristics portfolios

output = {'w_shift': pd.Series(w_shift),
          'h_char': pd.Series(h_char.reshape(t_ * n_)),
          'n_': pd.Series(n_)
          }
df = pd.DataFrame(output)
df.to_csv('../../../databases/temporary-databases/db_char_port.csv',
          index=None)

# ## Plots

# +
plt.style.use('arpm')

grid_dates = np.round(np.linspace(0, len(dates)-1, 5))

# P&L plot
fig1, ax1 = plt.subplots(1, 1)
ax2 = ax1.twinx()
ax1.set_ylabel('Cum P&L', color='b')
ax2.set_ylabel('P&L', color='r')
ax2.scatter(dates, z_char, color='r', linewidth=0, marker='.')
ax2.plot(dates, np.mean(z_char) + 2 * np.sqrt(sig2_pl_h_real), color='k')
ax2.plot(dates, np.mean(z_char) - 2 * np.sqrt(sig2_pl_h_real), color='k')

# cumulative P&L
cumpl = np.cumsum(z_char)
ax1.plot(dates, np.cumsum(z_char), color='b', lw=1.5)
ax1.axis([min(dates), max(dates), np.min(cumpl), np.max(cumpl)])
plt.title('Characteristic portfolio')
plt.xticks(dates[grid_dates.astype(int)])
fmt = mdates.DateFormatter('%d-%b-%y')
plt.gca().xaxis.set_major_formatter(fmt)

# expected returns vs realized returns
max_abs_ret = max(abs(np.percentile(real_ret, 100 * 0.05)),
                  abs(np.percentile(real_ret, 100 * 0.95)))

add_logo(fig1)

fig2 = plt.figure()
plt.plot(exp_ret, real_ret, 'b.')
plt.axis([np.min(exp_ret), np.max(exp_ret), -max_abs_ret, max_abs_ret])
plt.plot(x, y, 'r')
plt.xlabel('Expected returns')
plt.ylabel('Realized returns')

# signal
yy = np.linspace(1, n_, 5)
stock_tick = np.round(yy)
signal, idx = np.sort(s[-1, :]), np.argsort(s[-1, :])
max_abs_signal = max(abs(min(signal)), abs(max(signal)))

add_logo(fig2)

fig3 = plt.figure()
plt.bar(range(1, len(signal)+1), signal)
plt.ylabel('Signals')
plt.axis([0, exp_ret.shape[0] + 1, -max_abs_signal, max_abs_signal])
plt.xticks(stock_tick)

# exposures plot (sorted wrt the signals)
dollar_wghts = h_char[-1] * v[-2, :]

max_abs_dw = max(abs(min(dollar_wghts)), abs(max(dollar_wghts)))
add_logo(fig3)

fig4 = plt.figure()
plt.bar(range(1, len(dollar_wghts[idx])+1), dollar_wghts[idx])
plt.ylabel('Dollar weights')
plt.axis([0, exp_ret.shape[0] + 1, -max_abs_dw, max_abs_dw])
plt.xticks(stock_tick)
add_logo(fig4)

# premium
fig5 = plt.figure()
plt.plot(dates, lambda_hat, color='b')
plt.axis([min(dates), max(dates), np.nanmin(lambda_hat),
          np.nanmax(lambda_hat)])
plt.ylabel('Factor premium')
plt.xticks(dates[grid_dates.astype(int)])
myFmt = mdates.DateFormatter('%d-%b-%y')
plt.gca().xaxis.set_major_formatter(myFmt)
add_logo(fig5)
