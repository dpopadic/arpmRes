#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # s_pricing_couponbond [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_pricing_couponbond&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_couponbond).

# +
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

from arpym.pricing import bond_value, cash_flow_reinv, shadowrates_ytm
from arpym.statistics import meancov_sp
from arpym.tools import histogram_sp, add_logo

# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_couponbond-parameters)

tau_hor = 60  # time to horizon (months)
j_ = 1000  # number of scenarios
yields = True  # True if using yields, False if using shadow rates
c = 0.04  # annualized coupons (percentage of the face value)
freq_paym = 1  # coupon payment frequency (years)
value_plot = 1  # choose if visualizing the bond value
cashflow_plot = 1  # choose if visualizing the cash flow
pl_plot = 1  # choose if visualizing the P&L

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_couponbond-implementation-step00): Upload data

# +
# upload data from s_projection_yields_var1
path = '../../../databases/temporary-databases'
if yields:
    df = pd.read_csv(path + '/db_proj_scenarios_yield.csv', header=0)
    rd_type = 'y'
else:
    df = pd.read_csv(path + '/db_proj_scenarios_shadowrate.csv', header=0)
    rd_type = 'sr'
j_m_, _ = df.shape
df2 = pd.read_csv(path + '/db_proj_dates.csv', header=0, parse_dates=True)
t_m = np.array(pd.to_datetime(df2.values.reshape(-1)), dtype='datetime64[D]')
m_ = t_m.shape[0] - 1
deltat_m = np.busday_count(t_m[0], t_m[1])

if tau_hor > m_:
    print(" Projection doesn't have data until given horizon!!! Horizon lowered to ", m_)
    tau_hor = m_
# number of monitoring times
m_ = tau_hor
t_m = t_m[:m_ + 1]
tau = np.array(list(map(int, df.columns)))  # times to maturity
d_ = tau.shape[0]
x_tnow_thor = np.array(df).reshape(j_, int(j_m_ / j_), d_)
x_tnow_thor = x_tnow_thor[:j_, :m_ + 1, :]
t_m[-1]
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_couponbond-implementation-step01): Record dates and coupons of the bond

# +
# number of coupons until bond maturity
tend = np.datetime64('2022-06-29')  # bond time of maturity
k_ = int(np.busday_count(t_m[0], tend) / (freq_paym * 252))

# record dates
r = np.busday_offset(t_m[0], np.arange(1, k_ + 1) * int(freq_paym * 252))
# coupons
coupon = c * freq_paym * np.ones(k_)
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_couponbond-implementation-step02): Scenarios for bond value path

v_thor = np.array([bond_value(eval_t, coupon, r, rd_type, x_tnow_thor[:, m, :],
                              tau_x=tau)
                   for m, eval_t in enumerate(t_m)]).T

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_couponbond-implementation-step03): Scenarios for the reinvestment factor

# +
invfact_m = np.zeros((j_, m_))

for m in range(len(t_m[:-1])):
    interp = sp.interpolate.interp1d(tau.flatten(),
                                     x_tnow_thor[:, m, :],
                                     axis=1, fill_value='extrapolate')
    if yields:
        y_0 = interp(0)
    else:
        y_0 = shadowrates_ytm(interp(0))
    invfact_m[:, m] = np.exp(deltat_m * y_0 / 252)
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_couponbond-implementation-step04): Scenarios for the path of the cumulative cash flow

# +
# include notional with last coupon
coupon[-1] = coupon[-1] + 1

# cash flow streams path scenarios
cf_thor = cash_flow_reinv(coupon, r, t_m, invfact_m)
# -

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_couponbond-implementation-step05): Scenarios for the P&L of the bond

v_tnow = v_thor[:, 0].reshape(-1, 1)
pl_thor = v_thor - v_tnow + np.c_[np.zeros(j_), cf_thor]

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_couponbond-implementation-step06): Scenario-probability expectations and standard deviations

# +
mu_v_thor = np.zeros(m_ + 1)
sig_v_thor = np.zeros(m_ + 1)
mu_cf_thor = np.zeros(m_)
sig_cf_thor = np.zeros(m_)
mu_pl_thor = np.zeros(m_ + 1)
sig_pl_thor = np.zeros(m_ + 1)

# probabilities
for m in range(len(t_m)):
    mu_v_thor[m], sig1 = meancov_sp(v_thor[:, m].reshape(-1, 1))
    sig_v_thor[m] = np.sqrt(sig1)

    mu_pl_thor[m], sig1 = meancov_sp(pl_thor[:, m].reshape(-1, 1))
    sig_pl_thor[m] = np.sqrt(sig1)

for m in range(len(t_m) - 1):
    mu_cf_thor[m], sig1 = meancov_sp(cf_thor[:, m].reshape(-1, 1))
    sig_cf_thor[m] = np.sqrt(sig1)

# -

# ## [Step 7](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_couponbond-implementation-step07): Average yield scenarios at the horizon

if yields:
    y_bar = np.mean(shadowrates_ytm(x_tnow_thor), axis=2)
else:
    y_bar = np.mean(x_tnow_thor, axis=2)

# ## Plots

# +
plt.style.use('arpm')
lgrey = [0.8, 0.8, 0.8]  # light grey
dgrey = [0.4, 0.4, 0.4]  # dark grey
j_sel = 35  # selected MC simulations

# select what to plot among value, cumulative cash flow and P&L of the bond
y = []
mu = []
sig = []
obj = []
if value_plot:
    y.append(v_thor[:])
    mu.append(mu_v_thor[:])
    sig.append(sig_v_thor[:])
    obj.append('value')
if cashflow_plot:
    y.append(np.c_[np.zeros(j_), cf_thor[:]])
    mu.append(np.r_[0, mu_cf_thor[:]])
    sig.append(np.r_[0, sig_cf_thor[:]])
    obj.append('cash flow')
if pl_plot:
    y.append(pl_thor[:])
    mu.append(mu_pl_thor[:])
    sig.append(sig_pl_thor[:])
    obj.append('P&L')

fig_num = value_plot + cashflow_plot + pl_plot
for k in np.arange(fig_num):
    y_plot = np.array(y[k])
    mu_plot = np.array(mu[k])
    sig_plot = np.array(sig[k])
    obj_plot = obj[k]

    # simulated path, mean and standard deviation

    fig, axs = plt.subplots(2, 1)

    axs[0].set_position([0.05, 0.15, 0.65, 0.60])
    plt.sca(axs[0])
    t_axis = np.busday_count(t_m[0], t_m) / 252
    plt.plot(t_axis.reshape(-1, 1), y_plot[:j_sel, :].T, color=lgrey, lw=1)
    plt.yticks()
    plt.ylabel('Bond %s' % obj_plot)
    plt.xlabel('horizon')
    plt.xlim([np.min(t_axis), np.max(t_axis) + 3])
    l2 = plt.plot(t_axis, mu_plot + sig_plot, color='r')
    plt.plot(t_axis, mu_plot - sig_plot, color='r')
    l1 = plt.plot(t_axis, mu_plot, color='g')

    # empirical pdf
    p = np.ones(j_) / j_
    y_hist, x_hist = histogram_sp(y_plot[:, -1], k_=10 * np.log(j_))
    y_hist = y_hist / 10  # adapt the hist height to the current xaxis scale
    shift_y_hist = tau_hor / 12 + y_hist

    emp_pdf = plt.barh(x_hist, y_hist, left=t_axis[-1],
                       height=x_hist[1] - x_hist[0], facecolor=lgrey,
                       edgecolor=lgrey)

    plt.plot(shift_y_hist, x_hist, color=dgrey, lw=1)
    plt.plot([t_axis[-1], t_axis[-1]], [x_hist[0], x_hist[-1]], color=dgrey,
             lw=0.5)
    plt.legend(handles=[l1[0], l2[0], emp_pdf[0]],
               labels=['mean', ' + / - st.deviation', 'horizon pdf'])
    title = 'Coupon bond projected ' + obj_plot + ' at the horizon of ' + \
            str(tau_hor / 12) + ' years'
    plt.title(title)

    # scatter plot

    mydpi = 72.0
    axs[1].set_position([0.75, 0.25, 0.25, 0.40])
    plt.sca(axs[1])
    plt.xticks()
    plt.yticks()
    plt.scatter(y_bar[:, -1], y_plot[:, -1], 3, [dgrey], '*')
    plt.xlabel('Average yield')
    plt.ylabel('Coupon bond %s' % obj_plot)
    plt.title('Coupon bond %s vs. yields average' % obj_plot)

    add_logo(fig, axis=axs[0], size_frac_x=1 / 12)
# -




