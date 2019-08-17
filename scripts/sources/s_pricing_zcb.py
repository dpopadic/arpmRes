#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # s_pricing_zcb [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_pricing_zcb&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-pricing-normal-quad-approx).

# +
import numpy as np
import pandas as pd
from scipy.linalg import expm
import matplotlib.pyplot as plt
from datetime import timedelta

from arpym.pricing import zcb_value
from arpym.statistics import moments_mvou
from arpym.tools import histogram_sp, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_zcb-parameters)

tau_hor = 3    # time to horizon
j_ = 1000  # number of scenarios

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_zcb-implementation-step00): Upload data

# +
path = '../../../databases/temporary-databases'
df = pd.read_csv(path + '/db_proj_scenarios_yield.csv', header=0)

j_m_, _ = df.shape
df2 = pd.read_csv(path + '/db_proj_dates.csv', header=0, parse_dates=True)
t_m = np.array(pd.to_datetime(df2.values.reshape(-1)), dtype='datetime64[D]')
m_ = t_m.shape[0]-1
deltat_m = np.busday_count(t_m[0], t_m[1])

if tau_hor > m_:
    print(" Projection doesn't have data until given horizon!!! Horizon lowered to ", m_)
    tau_hor = m_
# number of monitoring times
m_ = tau_hor
t_m = t_m[:m_+1]
t_now = t_m[0]
t_hor = t_m[-1]
tau = np.array(list(map(int, df.columns)))  # times to maturity
d_ = tau.shape[0]
x_tnow_thor = np.array(df).reshape(j_, int(j_m_/j_), d_)
x_tnow_thor = x_tnow_thor[:j_, :m_+1, :]
y_tnow = x_tnow_thor[0, 0, :]
y_thor = x_tnow_thor[:, -1, :]

df = pd.read_csv(path + '/db_proj_scenarios_yield_par.csv', header=0)

theta = np.array(df['theta'].iloc[:d_ ** 2].values.reshape(d_, d_))
mu_mvou = np.array(df['mu_mvou'].iloc[:d_])
sig2_mvou = np.array(df['sig2_mvou'].iloc[:d_ ** 2].values.reshape(d_, d_))
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_zcb-implementation-step01): Compute zcb current values and scenarios for the value of the zcb at the horizon

# +
# zero-coupon bonds current values
t_end = np.array([np.datetime64('2012-10-24'), np.datetime64('2017-10-23')])  # zcb times of maturity
v_zcb_tnow = zcb_value(t_now, t_end, 'y', np.array([y_tnow]), tau_x=tau).squeeze()

# scenarios for zero-coupon bonds values at the horizon
v_zcb_thor = zcb_value(t_hor, t_end, 'y', y_thor, tau_x=tau).squeeze()
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_zcb-implementation-step02): Compute the zero-coupon bonds P&L's scenarios at the horizon

pl_thor = v_zcb_thor - v_zcb_tnow

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_zcb-implementation-step03): Compute the moments of the risk drivers at the horizon

mu_y, _, sig2_y = moments_mvou(y_tnow, [tau_hor*21],
                                     theta, mu_mvou, sig2_mvou)

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_zcb-implementation-step04): parameters for the first order Taylor approximation

# +
# numerical differentiation steps
dx = 0.001
dt = 1/252  # one day
n_ = len(v_zcb_tnow)  # number of instruments

y_up = np.tile(y_tnow.reshape(-1, 1), d_) + np.eye(d_) * dx
y_down = np.tile(y_tnow.reshape(-1, 1), d_) - np.eye(d_) * dx

# numerical yield
y_hat = 1 / (v_zcb_tnow * dt) * (zcb_value(t_now+np.timedelta64(int(dt*252)), t_end,
                                                 'y', np.array([y_tnow]), tau_x=tau) -
                                       v_zcb_tnow).squeeze()
# key rate durations
bond_t_up = zcb_value(t_now, t_end, 'y', y_up.T, tau_x=tau)
bond_t_down = zcb_value(t_now, t_end, 'y', y_down.T, tau_x=tau)
dur_hat = np.zeros((d_, n_))  # key-rates durations
for d in range(d_):
    dur_hat[d, :] = -(bond_t_up[d, :] - bond_t_down[d, :]) / \
                (v_zcb_tnow * 2 * dx)

# shift terms
deltat = tau_hor/12
alpha_pi_pric = y_hat * v_zcb_tnow *deltat

# exposures
beta_pi_pric = - dur_hat * v_zcb_tnow
# -

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_zcb-implementation-step05): parameters of the normal joint distribution of the P&L's

# +
mu_pl = alpha_pi_pric + \
    beta_pi_pric.T@((expm(-theta*deltat*252) -
                     np.eye(d_))@y_tnow +
                    mu_y)  # bonds' P&L's mean

sig2_pl = beta_pi_pric.T@sig2_y@beta_pi_pric  # bonds' P&L's covariance
# -

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_zcb-implementation-step06): Save database

output = {'j_': pd.Series(j_),
          'd_': pd.Series(d_),
          'time2hor_tnow': pd.Series(tau_hor/12),
          'time2mat_tnow': pd.Series([np.busday_count(t_now, t_end[i])/252
                                      for i in range(len(t_end))]),
          'tau_d': pd.Series(tau.reshape((d_,))),
          'pl1': pd.Series(pl_thor[:, 0].reshape((j_,))),
          'pl2': pd.Series(pl_thor[:, 1].reshape((j_,))),
          'v_zcb_tnow': pd.Series(v_zcb_tnow),
          'y_tnow': pd.Series(y_tnow),
          'theta': pd.Series(theta.reshape((d_ * d_,))),
          'mu_pl': pd.Series(mu_pl),
          'sig2_pl': pd.Series(sig2_pl.reshape((n_ * n_,))),
          'mu_thor': pd.Series(mu_y),
          'sig2_thor': pd.Series(sig2_y.reshape((d_ * d_,))),
          'dur_hat': pd.Series(dur_hat.reshape((d_ * n_,))),
          'y_hat': pd.Series(y_hat),
          'bond_t_up': pd.Series(bond_t_up.reshape((d_ * n_,))),
          'bond_t_down': pd.Series(bond_t_down.reshape((d_ * n_,))),
          'alpha_pi_pric': pd.Series(alpha_pi_pric.reshape((n_,))),
          'beta_pi_pric': pd.Series(beta_pi_pric.reshape((d_ * n_,)))
          }
df = pd.DataFrame(output)
df.to_csv('../../../databases/temporary-databases/db_pricing_zcb.csv')

# ## Plots

# +
plt.style.use('arpm')
n_ = sig2_pl.shape[1]
fig, ax = plt.subplots(n_, 1)

lgray = [.7, .7, .7]  # light gray
dgray = [.5, .5, .5]  # dark gray

for n in range(n_):
    # histogram of the zero coupon bond P&L
    plt.sca(ax[n])
    n_bins = round(15 * np.log(j_))  # number of histogram bins
    [f, x_f] = histogram_sp(pl_thor[:, [n]], p=(1/j_ * np.ones((j_, 1))),
                            k_=n_bins)
    hf = plt.bar(x_f, f, width=x_f[1] - x_f[0], facecolor=lgray,
                 edgecolor=dgray)
    if n == 0:
        plt.title(
            r'First zcb: distribution of the P&L at the horizon' +
            '\n' + r' $\tau$ = ' + str(tau_hor*21) + ' days')
    else:
        plt.title(r'Second zcb: distribution of the P&L at the horizon' +
                  '\n' + r' $\tau$ = ' +str(tau_hor*21) + ' days')
add_logo(fig, location=1)
plt.tight_layout()
